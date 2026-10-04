from __future__ import annotations

import datetime as dt

from tools.target_packs.hf_daily_papers import (
    CostScenario,
    date_window,
    estimate_costs,
    normalize_paper,
    normalize_window,
    unique_arxiv_records,
)


def test_date_window_is_inclusive() -> None:
    dates = date_window(dt.date(2026, 8, 16), 100)
    assert len(dates) == 100
    assert dates[0] == dt.date(2026, 5, 9)
    assert dates[-1] == dt.date(2026, 8, 16)


def test_normalize_paper_reads_nested_hf_shape() -> None:
    record = normalize_paper(
        "2026-08-16",
        1,
        {
            "paper": {
                "id": "2608.01234",
                "title": "Example",
                "summary": "Abstract",
                "authors": [{"name": "Ada"}, {"user": {"fullname": "Grace"}}],
                "upvotes": 42,
            }
        },
    )
    assert record["arxiv_id"] == "2608.01234"
    assert record["authors"] == ["Ada", "Grace"]
    assert record["daily_rank"] == 1
    assert record["upvotes_at_extraction"] == 42


def test_unique_arxiv_records_keeps_first_occurrence() -> None:
    rows = [
        {"arxiv_id": "2608.00002", "daily_rank": 2},
        {"arxiv_id": "2608.00001", "daily_rank": 1},
        {"arxiv_id": "2608.00002", "daily_rank": 1},
        {"arxiv_id": None, "daily_rank": 3},
    ]
    unique = unique_arxiv_records(rows)
    assert [row["arxiv_id"] for row in unique] == ["2608.00001", "2608.00002"]
    assert unique[1]["daily_rank"] == 2


def test_normalize_window_keeps_top_k_per_date() -> None:
    responses = {
        "2026-08-15": [{"paper": {"id": f"a.{i}"}} for i in range(12)],
        "2026-08-16": [{"paper": {"id": f"b.{i}"}} for i in range(4)],
    }
    records = normalize_window(responses, top_k=10, dataset_split="validation")
    assert len(records) == 14
    assert records[9]["daily_rank"] == 10
    assert records[10]["daily_rank"] == 1
    assert {record["dataset_split"] for record in records} == {"validation"}


def test_estimate_costs_includes_reasoning_as_output() -> None:
    estimate = estimate_costs(
        10,
        chars_per_token=4.0,
        prompt_overhead_tokens=1_000,
        scenarios=(CostScenario("test", 40_000, 2_000, 1_000),),
    )
    row = estimate["scenarios"][0]
    assert row["input_tokens_per_paper"] == 11_000
    assert row["billed_output_tokens_per_paper"] == 3_000
    standard = row["tiers"]["standard"]
    assert standard["total_cost_per_paper_usd"] == 0.145
    assert standard["total_cost_all_papers_usd"] == 1.45
