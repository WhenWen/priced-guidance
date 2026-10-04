"""Pilot title-only paper recall and LLM-judged summary similarity.

For each paper/model pair, ask the target model to summarize the paper from its
title alone.  A fixed judge then compares that recalled summary with a golden
summary produced from the paper's full LaTeX source.  Calls and outputs are
resumable at the per-paper level.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from tech_tree_arena.runtime import provider_client


DEFAULT_MODELS = (
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-5",
    "gpt-5.6-sol",
)
DEFAULT_JUDGE_MODEL = "gpt-5.6-terra"

MODEL_LABELS = {
    "anthropic/claude-fable-5": "Fable 5",
    "anthropic/claude-opus-5": "Opus 5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
}

RECALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
    },
    "required": ["summary"],
}

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "explanation": {"type": "string"},
    },
    "required": ["score", "explanation"],
}

RECALL_DEVELOPER = (
    "You write concise, technically substantive summaries of research papers. "
    "Return only the requested structured output."
)

JUDGE_DEVELOPER = (
    "You are a fixed evaluator of paper-summary similarity. Compare content, "
    "not prose style. Apply the same standard to every target model. Return only "
    "the requested structured output."
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty JSONL input: {path}")
    return rows


def _main_title_only(
    papers: list[dict[str, Any]], *, only_subtitled: bool
) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    for original in papers:
        full_title = str(original["title"])
        if ":" not in full_title:
            if only_subtitled:
                continue
            transformed.append(
                {
                    **original,
                    "full_title": full_title,
                    "subtitle": None,
                    "title_condition": "full_title_no_subtitle",
                }
            )
            continue
        main_title, subtitle = (part.strip() for part in full_title.split(":", 1))
        if not main_title or not subtitle:
            raise ValueError(f"invalid main-title/subtitle split: {full_title!r}")
        transformed.append(
            {
                **original,
                "title": main_title,
                "full_title": full_title,
                "subtitle": subtitle,
                "title_condition": "main_title_only",
            }
        )
    if not transformed:
        raise ValueError("main-title-only transform produced no papers")
    return transformed


def _load_gold(gold_dir: Path, arxiv_id: str) -> dict[str, Any]:
    path = gold_dir / "json" / f"{arxiv_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing golden summary: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    summary = record.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"gold record has no summary object: {path}")
    return summary


def _recall_user(title: str) -> str:
    return (
        "Summarize the research paper with the following title. Cover its research "
        "problem, proposed method, technical approach, and main findings.\n\n"
        f"PAPER TITLE:\n{title}"
    )


def _judge_user(title: str, recalled: str, golden: dict[str, Any]) -> str:
    return (
        "Compare the recalled summary with the golden summary. Rate how closely "
        "the recalled summary captures the paper's actual research problem, proposed "
        "method, technical approach, and main findings. Ignore differences in wording, "
        "organization, and level of detail. Return a similarity score from 0 to 100.\n\n"
        f"PAPER TITLE:\n{title}\n\n"
        f"RECALLED SUMMARY:\n{recalled}\n\n"
        "GOLDEN SUMMARY:\n"
        + json.dumps(golden, ensure_ascii=False, indent=2)
    )


def _recall_one(
    paper: dict[str, Any], model: str, out_dir: Path
) -> dict[str, Any]:
    arxiv_id = paper["arxiv_id"]
    path = out_dir / "recalls" / _slug(model) / f"{arxiv_id}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    result = provider_client.structured(
        model=model,
        developer=RECALL_DEVELOPER,
        user=_recall_user(paper["title"]),
        schema=RECALL_SCHEMA,
        schema_name="title_only_paper_recall",
        max_output_tokens=3_000,
        reasoning_effort="high",
        timeout=600.0,
    )
    record = {
        "arxiv_id": arxiv_id,
        "title": paper["title"],
        "full_title": paper.get("full_title", paper["title"]),
        "subtitle": paper.get("subtitle"),
        "short_name": paper.get("short_name"),
        "target_model": model,
        "prompt_condition": paper.get("title_condition", "full_title"),
        "recalled_summary": result["summary"],
    }
    _atomic_json(path, record)
    return record


def _judge_one(
    paper: dict[str, Any],
    model: str,
    judge_model: str,
    gold_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    arxiv_id = paper["arxiv_id"]
    path = out_dir / "judgments" / _slug(model) / f"{arxiv_id}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    recall = _recall_one(paper, model, out_dir)
    golden = _load_gold(gold_dir, arxiv_id)
    result = provider_client.structured(
        model=judge_model,
        developer=JUDGE_DEVELOPER,
        user=_judge_user(paper["title"], recall["recalled_summary"], golden),
        schema=JUDGE_SCHEMA,
        schema_name="paper_summary_similarity",
        max_output_tokens=2_000,
        reasoning_effort="high",
        timeout=600.0,
    )
    record = {
        "arxiv_id": arxiv_id,
        "title": paper["title"],
        "full_title": paper.get("full_title", paper["title"]),
        "subtitle": paper.get("subtitle"),
        "short_name": paper.get("short_name"),
        "target_model": model,
        "judge_model": judge_model,
        "score": result["score"],
        "explanation": result["explanation"],
    }
    _atomic_json(path, record)
    return record


def _aggregate(
    papers: list[dict[str, Any]],
    models: list[str],
    judge_model: str,
    out_dir: Path,
    provider_usage: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for paper in papers:
            path = out_dir / "judgments" / _slug(model) / f"{paper['arxiv_id']}.json"
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))

    csv_path = out_dir / "scores.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    csv_tmp.parent.mkdir(parents=True, exist_ok=True)
    with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "target_model",
            "judge_model",
            "arxiv_id",
            "short_name",
            "title",
            "full_title",
            "subtitle",
            "score",
            "explanation",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    csv_tmp.replace(csv_path)

    rows_by_key = {
        (str(row["target_model"]), str(row["arxiv_id"])): row for row in rows
    }
    matrix_path = out_dir / "score_matrix.csv"
    matrix_tmp = matrix_path.with_suffix(".csv.tmp")
    model_columns = [MODEL_LABELS.get(model, model) for model in models]
    with matrix_tmp.open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "arxiv_id",
            "full_title",
            "title_given",
            "title_condition",
            *model_columns,
            "mean",
            "spread",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for paper in papers:
            scores = [
                int(rows_by_key[(model, paper["arxiv_id"])]["score"])
                for model in models
                if (model, paper["arxiv_id"]) in rows_by_key
            ]
            matrix_row: dict[str, Any] = {
                "arxiv_id": paper["arxiv_id"],
                "full_title": paper.get("full_title", paper["title"]),
                "title_given": paper["title"],
                "title_condition": paper.get("title_condition", "full_title"),
                "mean": round(statistics.fmean(scores), 3) if scores else "",
                "spread": max(scores) - min(scores) if scores else "",
            }
            for model, column in zip(models, model_columns, strict=True):
                row = rows_by_key.get((model, paper["arxiv_id"]))
                matrix_row[column] = int(row["score"]) if row else ""
            writer.writerow(matrix_row)
    matrix_tmp.replace(matrix_path)

    by_model: dict[str, Any] = {}
    for model in models:
        scores = [int(row["score"]) for row in rows if row["target_model"] == model]
        by_model[model] = {
            "completed": len(scores),
            "mean": round(statistics.fmean(scores), 3) if scores else None,
            "median": statistics.median(scores) if scores else None,
            "minimum": min(scores) if scores else None,
            "maximum": max(scores) if scores else None,
        }
    cut_80_per_model = {
        model: sum(
            int(row["score"]) >= 80
            for row in rows
            if row["target_model"] == model
        )
        for model in models
    }
    complete_paper_scores = {
        paper["arxiv_id"]: [
            int(rows_by_key[(model, paper["arxiv_id"])]["score"])
            for model in models
            if (model, paper["arxiv_id"]) in rows_by_key
        ]
        for paper in papers
    }
    complete_paper_scores = {
        arxiv_id: scores
        for arxiv_id, scores in complete_paper_scores.items()
        if len(scores) == len(models)
    }
    report = {
        "status": "complete" if len(rows) == len(papers) * len(models) else "partial",
        "paper_count": len(papers),
        "target_models": models,
        "judge_model": judge_model,
        "title_conditions": sorted(
            {str(paper.get("title_condition", "full_title")) for paper in papers}
        ),
        "completed_judgments": len(rows),
        "expected_judgments": len(papers) * len(models),
        "score_summary": by_model,
        "cut_80": {
            "per_model": cut_80_per_model,
            "papers_any_model": sum(
                any(score >= 80 for score in scores)
                for scores in complete_paper_scores.values()
            ),
            "papers_all_models": sum(
                all(score >= 80 for score in scores)
                for scores in complete_paper_scores.values()
            ),
            "papers_mean": sum(
                statistics.fmean(scores) >= 80
                for scores in complete_paper_scores.values()
            ),
        },
        "provider_usage": provider_usage,
    }
    _atomic_json(out_dir / "report.json", report)
    return report


def _trace_usage(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return provider_client.usage_totals()
    calls = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    calls = [record for record in calls if record.get("kind") == "call"]
    by_model: dict[str, dict[str, Any]] = {}
    for record in calls:
        model = str(record.get("model") or "unknown")
        totals = by_model.setdefault(
            model,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )
        totals["calls"] += 1
        totals["input_tokens"] += int(record.get("input_tokens") or 0)
        totals["output_tokens"] += int(record.get("output_tokens") or 0)
        totals["cost_usd"] += float(record.get("cost_usd") or 0.0)
    for totals in by_model.values():
        totals["cost_usd"] = round(totals["cost_usd"], 8)
    return {
        "calls": len(calls),
        "input_tokens": sum(int(record.get("input_tokens") or 0) for record in calls),
        "output_tokens": sum(int(record.get("output_tokens") or 0) for record in calls),
        "cost_usd": round(sum(float(record.get("cost_usd") or 0.0) for record in calls), 8),
        "by_model": by_model,
    }


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", required=True, type=Path)
    parser.add_argument("--gold-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--trace", type=Path)
    parser.add_argument(
        "--strip-subtitles",
        action="store_true",
        help="Use only the text before the first colon as the prompt title.",
    )
    parser.add_argument(
        "--only-subtitled",
        action="store_true",
        help="With --strip-subtitles, run only papers whose full title contains a colon.",
    )
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.only_subtitled and not args.strip_subtitles:
        parser.error("--only-subtitled requires --strip-subtitles")
    if args.trace:
        provider_client.set_trace(args.trace)

    papers = _load_jsonl(args.papers)
    if args.strip_subtitles:
        papers = _main_title_only(papers, only_subtitled=args.only_subtitled)
    tasks = [(paper, model) for model in args.models for paper in papers]
    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _judge_one,
                paper,
                model,
                args.judge_model,
                args.gold_dir,
                args.out_dir,
            ): (paper, model)
            for paper, model in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            paper, model = futures[future]
            label = paper.get("short_name") or paper["arxiv_id"]
            try:
                row = future.result()
                print(f"{model} | {label}: {row['score']}")
            except Exception as exc:  # continue so a rerun resumes completed pairs
                failures.append(
                    {
                        "arxiv_id": paper["arxiv_id"],
                        "target_model": model,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"ERROR {model} | {label}: {type(exc).__name__}: {exc}")

    report = _aggregate(
        papers,
        args.models,
        args.judge_model,
        args.out_dir,
        _trace_usage(args.trace),
    )
    if failures:
        _atomic_json(args.out_dir / "last_run_failures.json", failures)
    elif (args.out_dir / "last_run_failures.json").exists():
        (args.out_dir / "last_run_failures.json").unlink()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
