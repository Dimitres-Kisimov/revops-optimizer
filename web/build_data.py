"""build_data.py — bundle the plan for the offline web dashboard.

Reads deliverables/prescription.json and writes web/data.js:

    window.PLAN = { ...the full plan... };
    window.PROMO_RESPONSE = { category: [a, b], ... };   // for the what-if slider
    window.PROMO_BUDGET = 40000;

and web/risk.js — the committed Monte-Carlo deliverables, passed through
verbatim so the dashboard can put the uncertainty band beside the headline €
(values are read from deliverables/uplift_simulation.csv and
deliverables/price_move_robustness.csv unchanged, never recomputed here):

    window.RISK = { metric: value, ... };                // uplift_simulation.csv
    window.GATE = [ { sku, ..., verdict, reason }, ... ];// price_move_robustness.csv
    window.GATE_RULE = { min_carry, min_agreement };     // the published gate

and web/compare.js — the committed A-vs-B comparison, likewise verbatim, so the
dashboard's compare panel shows the same bridge the CSV/SVG deliverables do:

    window.COMPARE = { summary: {metric: value, ...},    // plan_compare_summary.csv
                       bridge: [ {order, driver, ...} ], // plan_compare_bridge.csv
                       lines:  [ {kind, key, ...} ] };   // plan_compare_lines.csv

The promo response (a, b) parameters are pulled from the *same* function the
optimizer uses (`revops.optimize.promo._category_response`) so the client-side
"what-if" slider mirrors the real concave-allocation formula exactly.

    python -m revops.report        # writes deliverables/prescription.json
    python web/build_data.py       # writes web/data.js + risk.js + compare.js
                                   #   ->  open web/index.html

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:                    # allow `python web/build_data.py`
    sys.path.insert(0, str(_ROOT))
_PLAN = _ROOT / "deliverables" / "prescription.json"
_SIM_CSV = _ROOT / "deliverables" / "uplift_simulation.csv"
_GATE_CSV = _ROOT / "deliverables" / "price_move_robustness.csv"
_CMP_SUMMARY_CSV = _ROOT / "deliverables" / "plan_compare_summary.csv"
_CMP_BRIDGE_CSV = _ROOT / "deliverables" / "plan_compare_bridge.csv"
_CMP_LINES_CSV = _ROOT / "deliverables" / "plan_compare_lines.csv"
_OUT = Path(__file__).resolve().parent / "data.js"
_OUT_RISK = Path(__file__).resolve().parent / "risk.js"
_OUT_CMP = Path(__file__).resolve().parent / "compare.js"


def _num(s: str) -> int | float | str:
    """Parse a CSV cell without changing the number it states."""
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def _tall(path: Path) -> dict:
    """A committed (metric, value) CSV as a dict, verbatim."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {row["metric"]: _num(row["value"]) for row in csv.DictReader(f)}


def _rows(path: Path) -> list[dict]:
    """A committed fact-shaped CSV as a list of rows, verbatim."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _risk() -> dict:
    """The simulation summary, verbatim from the committed tall CSV."""
    return _tall(_SIM_CSV)


def _gate() -> list[dict]:
    """The per-move accept/hold rows, verbatim from the committed CSV."""
    return _rows(_GATE_CSV)


def _compare() -> dict:
    """The committed A-vs-B comparison — summary, bridge and line diff — passed
    through unchanged, so the panel and the CSV/SVG deliverables cannot drift."""
    return {"summary": _tall(_CMP_SUMMARY_CSV),
            "bridge": _rows(_CMP_BRIDGE_CSV),
            "lines": _rows(_CMP_LINES_CSV)}


def _gate_rule() -> dict:
    """The published gate thresholds, from the module that owns them."""
    try:
        from revops.robustness import ACCEPT_MIN_AGREEMENT, ACCEPT_MIN_CARRY
        return {"min_carry": ACCEPT_MIN_CARRY, "min_agreement": ACCEPT_MIN_AGREEMENT}
    except Exception:
        return {}


def _promo_response(plan: dict) -> dict[str, list[float]]:
    """(a, b) per carried category, from the optimizer's own response model."""
    try:
        from revops.dataio import load
        from revops.optimize.promo import _category_response
        carried = set(plan["plan"]["assortment"]["carried"])
        skus = [s for s in load() if s.sku in carried]
        params = _category_response(skus)
        return {c: [round(float(a), 4), round(float(b), 8)]
                for c, (a, b) in params.items()}
    except Exception:
        return {}


def _assort_by_cat(plan: dict) -> dict[str, dict]:
    """Per-category carried/dropped annual margin — for the assortment bar."""
    try:
        from revops.dataio import load
    except Exception:
        return {}
    carried = set(plan["plan"]["assortment"]["carried"])
    out: dict[str, dict] = {}
    for s in load():
        d = out.setdefault(s.category, {"carried_margin": 0.0,
                                        "dropped_margin": 0.0,
                                        "n_carried": 0, "n_dropped": 0})
        if s.sku in carried:
            d["carried_margin"] += s.annual_margin_eur
            d["n_carried"] += 1
        else:
            d["dropped_margin"] += s.annual_margin_eur
            d["n_dropped"] += 1
    for d in out.values():
        d["carried_margin"] = round(d["carried_margin"], 2)
        d["dropped_margin"] = round(d["dropped_margin"], 2)
    return out


def build() -> Path:
    if not _PLAN.exists():
        raise SystemExit(
            f"{_PLAN} not found — run `python -m revops.report` first.")
    with _PLAN.open(encoding="utf-8") as f:
        plan = json.load(f)

    resp = _promo_response(plan)
    assort = _assort_by_cat(plan)
    budget = plan["plan"]["promo"]["budget_eur"]

    with _OUT.open("w", encoding="utf-8") as f:
        f.write("// Auto-generated by web/build_data.py — do not edit by hand.\n")
        f.write("window.PLAN = ")
        json.dump(plan, f, indent=1)
        f.write(";\n")
        f.write("window.PROMO_RESPONSE = ")
        json.dump(resp, f)
        f.write(";\n")
        f.write("window.ASSORT_BY_CAT = ")
        json.dump(assort, f)
        f.write(";\n")
        f.write(f"window.PROMO_BUDGET = {budget};\n")

    with _OUT_RISK.open("w", encoding="utf-8") as f:
        f.write("// Auto-generated by web/build_data.py — do not edit by hand.\n")
        f.write("// Values are the committed Monte-Carlo deliverables, verbatim.\n")
        f.write("window.RISK = ")
        json.dump(_risk(), f, indent=1)
        f.write(";\n")
        f.write("window.GATE = ")
        json.dump(_gate(), f, indent=1)
        f.write(";\n")
        f.write("window.GATE_RULE = ")
        json.dump(_gate_rule(), f)
        f.write(";\n")

    with _OUT_CMP.open("w", encoding="utf-8") as f:
        f.write("// Auto-generated by web/build_data.py — do not edit by hand.\n")
        f.write("// Values are the committed plan-compare deliverables, verbatim.\n")
        f.write("window.COMPARE = ")
        json.dump(_compare(), f, indent=1)
        f.write(";\n")
    return _OUT


if __name__ == "__main__":
    p = build()
    kb = p.stat().st_size / 1024
    kb_r = _OUT_RISK.stat().st_size / 1024
    kb_c = _OUT_CMP.stat().st_size / 1024
    print(f"  ok data.js    ({kb:.0f} KB) -> {p.relative_to(_ROOT)}")
    print(f"  ok risk.js    ({kb_r:.0f} KB) -> {_OUT_RISK.relative_to(_ROOT)}")
    print(f"  ok compare.js ({kb_c:.0f} KB) -> {_OUT_CMP.relative_to(_ROOT)}")
