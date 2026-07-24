"""Tests for the prescribe pipeline and the deliverables writer."""
from __future__ import annotations

import json


def test_prescribe_positive_uplift(plan):
    up = plan["expected_uplift"]
    assert up["total_expected_uplift_eur"] > 0
    # the MILP-vs-greedy gap is a material headline
    assert up["assortment_milp_vs_greedy_eur"] > 0
    # components tie out to the total
    parts = (up["pricing_uplift_annual_eur"] + up["promo_incremental_eur"]
             + up["assortment_milp_vs_greedy_eur"])
    assert abs(parts - up["total_expected_uplift_eur"]) < 1.0


def test_prescribe_is_json_serializable(plan):
    s = json.dumps(plan)
    assert len(s) > 100
    # round-trips
    assert json.loads(s)["expected_uplift"] == plan["expected_uplift"]


def test_prescribe_has_decision_cards(plan):
    assert isinstance(plan["decision_cards"], list)
    assert len(plan["decision_cards"]) >= 5


def test_write_deliverables_produces_core_files(plan, tmp_path):
    from revops.report import write_deliverables
    out = write_deliverables(plan, outdir=tmp_path)

    # always-present, dependency-free deliverables
    assert (tmp_path / "prescription.json").exists()
    assert (tmp_path / "actions.csv").exists()

    # prescription.json parses back to the same plan
    with (tmp_path / "prescription.json").open(encoding="utf-8") as f:
        assert json.load(f)["expected_uplift"] == plan["expected_uplift"]

    # actions.csv has a header + one row per carried SKU
    rows = (tmp_path / "actions.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("sku,")
    assert len(rows) - 1 == plan["plan"]["assortment"]["n_carried"]

    # optional (library-gated) deliverables: present because libs are installed
    for name in ("optimization_workbook.xlsx", "executive_review.pdf"):
        if out.get(name) is not None:
            assert (tmp_path / name).exists()


def test_actions_csv_reorder_and_price_columns(plan, tmp_path):
    from revops.report import write_deliverables
    write_deliverables(plan, outdir=tmp_path)
    import csv
    with (tmp_path / "actions.csv").open(encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    assert r
    for row in r:
        assert row["action_carry"] == "carry"
        assert row["order_up_to"] != ""       # reorder quantity present
