"""forecast.py — a GLOBAL demand forecaster (one model across all SKUs).

Following the M5 / DeepAR lesson that a single global model on lag + calendar
features beats fitting a tiny model per short series, we:

  * engineer, per (sku, month): lags 1/2/3/12, a rolling mean(3), month Fourier
    terms (sin/cos, k=1,2) and a per-SKU scale — everything expressed in a
    scale-free way (units divided by the SKU's own level) so one network
    generalizes across a €0.9 fastener and a €115 power tool;
  * train a small PyTorch MLP (10 -> 32 -> 16 -> 1, ReLU, dropout 0.1, Adam,
    ~40 epochs, MSE on log1p of the scaled demand), printing a shrinking
    train/val loss;
  * backtest with rolling-origin one-step forecasts and report MASE against a
    seasonal-naive (units[t-12]) baseline — reported honestly.

Public API:
    train(history=None) -> ForecastModel
    forecast_next(model, sku_history) -> float
    backtest(history=None) -> dict           # MASE + baseline
    build_forecast_cache(...) -> Path         # writes data/forecasts.csv

Weights are saved to models/forecast.pt (gitignored); the per-SKU next-month
forecast is cached to data/forecasts.csv for the optimizers to consume.

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
_HISTORY = _ROOT / "data" / "demand_history.csv"
_FORECAST_CSV = _ROOT / "data" / "forecasts.csv"
_WEIGHTS = _ROOT / "models" / "forecast.pt"

SEED = 1234
N_LAGS = (1, 2, 3, 12)
N_FEATURES = 10          # 4 lags + roll3 + sin1 cos1 sin2 cos2 + log1p(scale)


# --------------------------------------------------------------------------- #
# data loading                                                                #
# --------------------------------------------------------------------------- #
def load_history(path: str | Path | None = None) -> dict[str, list[dict]]:
    """Return {sku: [rows in month order]} from the history CSV."""
    p = Path(path) if path else _HISTORY
    series: dict[str, list[dict]] = {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            series.setdefault(r["sku"], []).append({
                "month": r["month"],
                "price": float(r["price_eur"]),
                "units": float(r["units"]),
                "baseline": float(r["demand_baseline"]),
            })
    for rows in series.values():
        rows.sort(key=lambda x: x["month"])
    return series


def _cal_month(label: str) -> int:
    return int(label.split("-")[1])


def _fourier(cal_m: int) -> list[float]:
    ang = 2 * math.pi * (cal_m - 1) / 12
    return [math.sin(ang), math.cos(ang), math.sin(2 * ang), math.cos(2 * ang)]


def _features_at(units: list[float], labels: list[str], t: int,
                 scale: float) -> list[float] | None:
    """Feature row for predicting units[t]; None if a required lag is missing."""
    if t < max(N_LAGS):
        return None
    lags = [units[t - k] / scale for k in N_LAGS]
    roll3 = float(np.mean(units[t - 3:t])) / scale
    fou = _fourier(_cal_month(labels[t]))
    return lags + [roll3, *fou, math.log1p(scale)]


def _scale_of(units: list[float], upto: int | None = None) -> float:
    """Per-SKU level = mean of (a leading window of) the series. >0."""
    window = units[:upto] if upto else units
    window = window[:12] if len(window) >= 12 else window
    return max(1e-3, float(np.mean(window)))


def _build_dataset(series: dict[str, list[dict]], t_max: int | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Global (X, y) over every SKU and every month with full lags.
    y = log1p(units[t] / scale); features are scale-free. t_max caps the last
    usable month index (exclusive-inclusive: use t in [12, t_max])."""
    X, y = [], []
    for rows in series.values():
        units = [r["units"] for r in rows]
        labels = [r["month"] for r in rows]
        scale = _scale_of(units)
        hi = (t_max if t_max is not None else len(units) - 1)
        for t in range(max(N_LAGS), hi + 1):
            feat = _features_at(units, labels, t, scale)
            if feat is None:
                continue
            X.append(feat)
            y.append(math.log1p(units[t] / scale))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)


# --------------------------------------------------------------------------- #
# model                                                                        #
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, n_in: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class ForecastModel:
    net: _MLP
    feat_mean: np.ndarray
    feat_std: np.ndarray

    def _predict_scaled(self, feats: np.ndarray) -> np.ndarray:
        z = (feats - self.feat_mean) / self.feat_std
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(z, dtype=torch.float32)).numpy()
        return out                                   # log1p(units/scale)


def _fit(X: np.ndarray, y: np.ndarray, epochs: int = 40, verbose: bool = True
         ) -> ForecastModel:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xz = (X - mean) / std

    idx = rng.permutation(len(Xz))
    n_val = max(1, int(0.2 * len(idx)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr = torch.tensor(Xz[tr_idx], dtype=torch.float32)
    ytr = torch.tensor(y[tr_idx], dtype=torch.float32)
    Xva = torch.tensor(Xz[val_idx], dtype=torch.float32)
    yva = torch.tensor(y[val_idx], dtype=torch.float32)

    net = _MLP()
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(Xtr), ytr)
        loss.backward()
        opt.step()
        if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
            net.eval()
            with torch.no_grad():
                vl = loss_fn(net(Xva), yva).item()
            print(f"  epoch {ep:>3}/{epochs}  train_mse={loss.item():.4f}  "
                  f"val_mse={vl:.4f}")

    return ForecastModel(net=net, feat_mean=mean, feat_std=std)


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #
def train(history: dict[str, list[dict]] | None = None,
          save: bool = True, verbose: bool = True) -> ForecastModel:
    """Train the global forecaster on all available months."""
    series = history if history is not None else load_history()
    X, y = _build_dataset(series)
    if verbose:
        print(f"[forecast] training global MLP on {len(X)} rows "
              f"({len(series)} SKUs, {N_FEATURES} features)")
    model = _fit(X, y, verbose=verbose)
    if save:
        _WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.net.state_dict(),
                    "mean": model.feat_mean, "std": model.feat_std}, _WEIGHTS)
    return model


def forecast_next(model: ForecastModel, sku_history: list[dict]) -> float:
    """One-month-ahead demand forecast for a single SKU's history rows."""
    units = [r["units"] for r in sku_history]
    labels = [r["month"] for r in sku_history]
    scale = _scale_of(units)

    # feature row for the *next* month (index = len(units)); needs synthetic
    # calendar label but we only use the calendar month for Fourier terms.
    last = labels[-1]
    y, m = int(last[:4]), int(last[5:7])
    m += 1
    if m > 12:
        m, y = 1, y + 1
    next_label = f"{y:04d}-{m:02d}"
    ext_units = units + [0.0]              # placeholder; not read for lags/roll
    ext_labels = labels + [next_label]
    t = len(units)                          # predicting this index
    feat = _features_at(ext_units, ext_labels, t, scale)
    if feat is None:
        return float(np.mean(units[-3:]) if units else 0.0)
    scaled = model._predict_scaled(np.asarray([feat], dtype=np.float64))[0]
    return float(max(0.0, math.expm1(scaled) * scale))


def backtest(history: dict[str, list[dict]] | None = None,
             horizon: int = 3, verbose: bool = False) -> dict:
    """Rolling-origin one-step backtest. Train on months up to (T-horizon),
    then teacher-forced one-step forecasts for the final `horizon` months.
    MASE = model MAE / in-sample seasonal-naive MAE (units[t-12])."""
    series = history if history is not None else load_history()
    T = min(len(r) for r in series.values())
    cutoff = T - 1 - horizon                       # last training month index

    X, y = _build_dataset(series, t_max=cutoff)
    model = _fit(X, y, verbose=verbose)

    abs_err_model, abs_err_naive = [], []
    naive_insample = []
    for rows in series.values():
        units = [r["units"] for r in rows]
        labels = [r["month"] for r in rows]
        scale = _scale_of(units, upto=cutoff + 1)

        # in-sample seasonal-naive errors (the MASE denominator)
        for t in range(12, cutoff + 1):
            naive_insample.append(abs(units[t] - units[t - 12]))

        for t in range(cutoff + 1, T):
            feat = _features_at(units, labels, t, scale)
            if feat is None:
                continue
            pred = math.expm1(
                model._predict_scaled(np.asarray([feat]))[0]) * scale
            abs_err_model.append(abs(units[t] - max(0.0, pred)))
            abs_err_naive.append(abs(units[t] - units[t - 12]))   # seasonal naive

    denom = float(np.mean(naive_insample)) if naive_insample else 1.0
    denom = denom if denom > 1e-9 else 1.0
    model_mae = float(np.mean(abs_err_model))
    naive_mae = float(np.mean(abs_err_naive))
    return {
        "mase": round(model_mae / denom, 4),
        "naive_mase": round(naive_mae / denom, 4),
        "model_mae": round(model_mae, 3),
        "seasonal_naive_mae": round(naive_mae, 3),
        "horizon_months": horizon,
        "n_forecasts": len(abs_err_model),
        "beats_naive": model_mae < naive_mae,
    }


def build_forecast_cache(model: ForecastModel | None = None,
                         history: dict[str, list[dict]] | None = None,
                         path: str | Path | None = None) -> Path:
    """Compute next-month demand per SKU and write data/forecasts.csv."""
    series = history if history is not None else load_history()
    model = model if model is not None else train(series, verbose=False)
    out = Path(path) if path else _FORECAST_CSV
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sku", "forecast_next_month"])
        for sku, rows in series.items():
            w.writerow([sku, round(forecast_next(model, rows), 2)])
    return out


def load_forecast_cache(path: str | Path | None = None) -> dict[str, float]:
    p = Path(path) if path else _FORECAST_CSV
    if not p.exists():
        return {}
    out = {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[r["sku"]] = float(r["forecast_next_month"])
    return out


if __name__ == "__main__":
    hist = load_history()
    m = train(hist)
    print("[forecast] backtest:", backtest(hist))
    p = build_forecast_cache(m, hist)
    print(f"[forecast] cached next-month forecasts -> {p}")
