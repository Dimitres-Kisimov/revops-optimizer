"""revops CLI — run the full predict -> optimize -> prescribe pipeline.

    python -m revops                 # headline summary + decision cards
    python -m revops --json out.json # dump the full prescription to JSON
    python -m revops --budget 80000 --promo-budget 50000 --service-level 0.98

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import argparse
import json
import sys

from .prescribe import prescribe


def _utf8_stdout() -> None:
    """Make stdout UTF-8 safe on Windows terminals (cp1252 default)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")          # py3.7+
    except Exception:
        pass


def _headline(p: dict) -> str:
    up = p["expected_uplift"]
    a = p["plan"]["assortment"]
    pr = p["plan"]["pricing"]
    pm = p["plan"]["promo"]
    fq = p["model_quality"]["forecast"]
    eq = p["model_quality"]["elasticity"]
    rq = p["model_quality"]["decline_risk"]

    alloc = sorted(pm["allocation"].items(), key=lambda kv: -kv[1])
    promo_split = "  ".join(f"{c}=EUR{v:,.0f}" for c, v in alloc if v > 1)

    lines = [
        "=" * 68,
        " REVOPS — prescriptive plan (AI forecasts -> optimization core)",
        "=" * 68,
        f" Expected uplift        EUR {up['total_expected_uplift_eur']:>12,.0f} / year",
        f"   - pricing            EUR {up['pricing_uplift_annual_eur']:>12,.0f}",
        f"   - promo              EUR {up['promo_incremental_eur']:>12,.0f}",
        f"   - assortment (MILP)  EUR {up['assortment_uplift_vs_greedy_eur']:>12,.0f}",
        "-" * 68,
        f" SKUs carried           {a['n_carried']} of {a['n_carried'] + a['n_dropped']}"
        f"   (capital EUR {a['capital_used_eur']:,.0f} / {a['budget_eur']:,.0f})",
        f" MILP vs greedy         EUR {a['milp_uplift_vs_greedy_eur']:>12,.0f} / year"
        f"   (two constraints: capital + shelf)",
        f" Price moves            {pr['n_moves']} SKUs repriced within guardrail",
        f" Promo split            {promo_split}",
        "-" * 68,
        " Model quality (honest, held-out):",
        f"   Forecast MASE        {fq['mase']}  (seasonal-naive {fq['naive_mase']}; "
        f"{'beats' if fq['beats_naive'] else 'below'} naive)",
        f"   Elasticity recovery  MAE {eq['mae_by_category']} by category, "
        f"{eq['mae_by_sku']} by SKU",
        f"   Decline-risk AUC     ROC {rq['roc_auc']}  PR {rq['pr_auc']}  "
        f"(prevalence {rq['prevalence']})",
        "=" * 68,
        " Decision cards:",
    ]
    for i, card in enumerate(p["decision_cards"], 1):
        lines.append(f"  {i}. {card}")
    lines.append("=" * 68)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(prog="revops", description=__doc__)
    ap.add_argument("--budget", type=float, default=60_000.0,
                    help="working-capital budget for the assortment MILP")
    ap.add_argument("--shelf-capacity-m3", type=float, default=10.0,
                    help="shelf-space budget (m3) — the 2nd MILP knapsack dim")
    ap.add_argument("--service-level", type=float, default=0.95)
    ap.add_argument("--promo-budget", type=float, default=40_000.0)
    ap.add_argument("--price-guardrail", type=float, default=0.15)
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="write the full prescription to this JSON file")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the training loss printout")
    args = ap.parse_args(argv)

    plan = prescribe(
        budget=args.budget, shelf_capacity_m3=args.shelf_capacity_m3,
        service_level=args.service_level, promo_budget=args.promo_budget,
        price_guardrail=args.price_guardrail, verbose=not args.quiet)

    print(_headline(plan))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
        print(f"\nFull prescription written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
