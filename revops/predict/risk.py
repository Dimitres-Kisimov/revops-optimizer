"""risk.py — SKU demand-decline classifier (from-scratch logistic regression).

We label a SKU "declining" when the slope of its demand over the 24-month
history is materially negative (normalized by its own level), then train an
L2-penalized, class-weighted logistic regression **from scratch in numpy**
(sigmoid link, binary cross-entropy, analytic gradient, full-batch gradient
descent) to predict that label from four cheap features:

    * recent-vs-older demand ratio  (mean of last 6 mo / mean of first 6 mo)
    * normalized trend slope         (OLS slope / mean level)
    * coefficient of variation       (demand volatility)
    * margin_pct                     (commercial quality)

Quality is reported with from-scratch ROC-AUC and PR-AUC on a held-out split.
`score_decline_risk` trains on all SKUs and returns a decline probability each,
which the prescriber uses to down-weight fragile SKUs in the assortment.

    score_decline_risk(history) -> dict[sku, float]
    quality_report(history) -> dict

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import numpy as np

from ..dataio import load
from .forecast import load_history

SEED = 99
DECLINE_THRESHOLD = -0.010          # slope/level below this => "declining"


# --------------------------------------------------------------------------- #
# feature / label engineering                                                  #
# --------------------------------------------------------------------------- #
def _sku_features_label(units: list[float], margin_pct: float
                        ) -> tuple[list[float], int, float]:
    u = np.asarray(units, dtype=np.float64)
    t = np.arange(len(u), dtype=np.float64)
    level = max(1e-6, float(u.mean()))

    # OLS slope of units on month index
    slope = float(np.polyfit(t, u, 1)[0])
    norm_slope = slope / level

    half = len(u) // 2
    older = max(1e-6, float(u[:half].mean()))
    recent = float(u[half:].mean())
    ratio = recent / older
    cv = float(u.std()) / level

    label = 1 if norm_slope < DECLINE_THRESHOLD else 0
    feats = [ratio, norm_slope, cv, margin_pct]
    return feats, label, norm_slope


def _build_matrix(history: dict[str, list[dict]]
                  ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    skus = load()
    margin_of = {s.sku: s.margin_pct for s in skus}
    X, y, order = [], [], []
    for sku, rows in history.items():
        if sku not in margin_of:
            continue
        feats, label, _ = _sku_features_label([r["units"] for r in rows],
                                              margin_of[sku])
        X.append(feats)
        y.append(label)
        order.append(sku)
    return np.asarray(X, float), np.asarray(y, float), order


# --------------------------------------------------------------------------- #
# logistic regression (from scratch)                                           #
# --------------------------------------------------------------------------- #
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogReg:
    def __init__(self, l2: float = 1e-2, lr: float = 0.3, epochs: int = 800):
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sd = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogReg:
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        Xz = (X - self.mu) / self.sd
        n, d = Xz.shape

        # class weights for imbalance: weight_c = n / (2 * n_c)
        pos = max(1.0, y.sum())
        neg = max(1.0, n - y.sum())
        w_pos, w_neg = n / (2 * pos), n / (2 * neg)
        sample_w = np.where(y == 1, w_pos, w_neg)

        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = _sigmoid(Xz @ self.w + self.b)
            err = (p - y) * sample_w
            grad_w = Xz.T @ err / n + self.l2 * self.w      # analytic gradient
            grad_b = float(err.sum() / n)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xz = (X - self.mu) / self.sd
        return _sigmoid(Xz @ self.w + self.b)


# --------------------------------------------------------------------------- #
# metrics (from scratch)                                                        #
# --------------------------------------------------------------------------- #
def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC via the rank (Mann-Whitney U) identity."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _tie_average(s, ranks)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _tie_average(s: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(s, kind="mergesort")
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s[order[j + 1]] == s[order[i]]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision = area under the precision-recall curve."""
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(1, tp + fp)
    total_pos = max(1.0, y.sum())
    recall = tp / total_pos
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall, strict=False):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


# --------------------------------------------------------------------------- #
# public API                                                                    #
# --------------------------------------------------------------------------- #
def score_decline_risk(history: dict[str, list[dict]] | None = None
                       ) -> dict[str, float]:
    hist = history if history is not None else load_history()
    X, y, order = _build_matrix(hist)
    model = LogReg().fit(X, y)
    proba = model.predict_proba(X)
    return {sku: round(float(p), 4) for sku, p in zip(order, proba, strict=False)}


def quality_report(history: dict[str, list[dict]] | None = None) -> dict:
    """Stratified holdout (70/30) test AUCs — honest generalization estimate."""
    hist = history if history is not None else load_history()
    X, y, _ = _build_matrix(hist)
    rng = np.random.default_rng(SEED)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_pos_te = max(1, int(0.3 * len(pos_idx)))
    n_neg_te = max(1, int(0.3 * len(neg_idx)))
    test_idx = np.concatenate([pos_idx[:n_pos_te], neg_idx[:n_neg_te]])
    train_idx = np.concatenate([pos_idx[n_pos_te:], neg_idx[n_neg_te:]])

    model = LogReg().fit(X[train_idx], y[train_idx])
    s_te = model.predict_proba(X[test_idx])
    return {
        "roc_auc": round(roc_auc(y[test_idx], s_te), 4),
        "pr_auc": round(pr_auc(y[test_idx], s_te), 4),
        "n_declining": int(y.sum()),
        "n_total": int(len(y)),
        "prevalence": round(float(y.mean()), 3),
        "n_test": int(len(test_idx)),
    }


if __name__ == "__main__":
    rep = quality_report()
    print(f"[risk] declining SKUs: {rep['n_declining']}/{rep['n_total']} "
          f"(prevalence {rep['prevalence']})")
    print(f"[risk] test ROC-AUC = {rep['roc_auc']}   PR-AUC = {rep['pr_auc']}")
