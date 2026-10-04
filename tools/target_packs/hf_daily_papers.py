"""Extract Hugging Face Daily Papers over a date window and estimate summary cost.

The script fetches every Daily Papers occurrence, retains the top K from each
date for the training set, and writes the original per-date API responses, a
de-duplicated arXiv id list, daily counts, and GPT-5.6 Sol cost scenarios for
producing the repository's golden summaries.

Examples:

    python tools/target_packs/hf_daily_papers.py \
      --days 100 \
      --end-date 2026-08-16 \
      --top-k 10 \
      --out-dir artifacts/hf-daily-papers-top10-100d-2026-08-16

    # Estimate without fetching, using an existing extraction.
    python tools/target_packs/hf_daily_papers.py \
      --estimate-only \
      --out-dir artifacts/hf-daily-papers-top10-100d-2026-08-16

The cost model is deliberately separate from generation. It never calls OpenAI.
Pricing is a dated snapshot from the official OpenAI pricing page and can be
updated in the constants near the top of this file when prices change.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


HF_API_BASE = "https://huggingface.co/api/daily_papers"
HF_API_TEMPLATE = HF_API_BASE + "?date={date}&limit={limit}&p={page}"
CACHE_SCHEMA_VERSION = 2
DEFAULT_PAGE_SIZE = 100
OPENAI_PRICING_URL = "https://developers.openai.com/api/docs/pricing"
MODEL = "gpt-5.6-sol"
PRICING_AS_OF = "2026-08-16"
SHORT_CONTEXT_LIMIT = 272_000
DEFAULT_MAX_SOURCE_CHARS = 140_000
DEFAULT_CHARS_PER_TOKEN = 4.0
DEFAULT_PROMPT_OVERHEAD_TOKENS = 3_500


@dataclass(frozen=True, slots=True)
class PriceTier:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


PRICES: dict[str, PriceTier] = {
    "standard": PriceTier(5.00, 0.50, 30.00),
    "batch": PriceTier(2.50, 0.25, 15.00),
    "flex": PriceTier(2.50, 0.25, 15.00),
    "fast": PriceTier(10.00, 1.00, 60.00),
}


@dataclass(frozen=True, slots=True)
class CostScenario:
    name: str
    source_chars_per_paper: int
    reasoning_tokens_per_paper: int
    visible_output_tokens_per_paper: int


DEFAULT_SCENARIOS = (
    CostScenario("low", 60_000, 1_500, 1_500),
    CostScenario("base", 100_000, 5_000, 2_000),
    CostScenario("high", 140_000, 10_000, 4_000),
)


def date_window(end_date: dt.date, days: int) -> tuple[dt.date, ...]:
    if days < 1:
        raise ValueError("days must be at least 1")
    start = end_date - dt.timedelta(days=days - 1)
    return tuple(start + dt.timedelta(days=offset) for offset in range(days))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _fetch_date(
    date: dt.date,
    *,
    timeout: float,
    retries: int,
    page_size: int,
    raw_dir: Path,
    refresh: bool,
) -> tuple[str, list[dict[str, Any]]]:
    date_text = date.isoformat()
    cache_path = raw_dir / f"{date_text}.json"
    cache_meta_path = raw_dir / f"{date_text}.meta.json"
    if cache_path.is_file() and cache_meta_path.is_file() and not refresh:
        cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        value = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cache_meta.get("schema_version") != CACHE_SCHEMA_VERSION
            or cache_meta.get("page_size") != page_size
        ):
            value = None
        if value is not None and not isinstance(value, list):
            raise ValueError(f"cached response for {date_text} is not an array")
        if value is not None:
            return date_text, value

    papers: list[dict[str, Any]] = []
    page_lengths: list[int] = []
    page = 0
    while True:
        query = urllib.parse.urlencode(
            {"date": date_text, "limit": page_size, "p": page}
        )
        url = f"{HF_API_BASE}?{query}"
        last_error: Exception | None = None
        page_value: list[dict[str, Any]] | None = None
        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "idea-arena-hf-daily-papers/1.0",
                    },
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    value = json.load(response)
                if not isinstance(value, list):
                    raise ValueError(
                        f"response for {date_text} page {page} is not an array"
                    )
                page_value = value
                break
            except (
                OSError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
            ) as exc:
                last_error = exc
                if attempt == retries:
                    break
                time.sleep(min(2**attempt, 8))
        if page_value is None:
            raise RuntimeError(
                f"failed to fetch {date_text} page {page}: {last_error}"
            )

        page_lengths.append(len(page_value))
        papers.extend(page_value)
        if len(page_value) < page_size:
            break
        page += 1

    _atomic_write_text(
        cache_path,
        json.dumps(papers, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        cache_meta_path,
        json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "date": date_text,
                "page_size": page_size,
                "page_lengths": page_lengths,
                "paper_count": len(papers),
            },
            indent=2,
        )
        + "\n",
    )
    return date_text, papers


def fetch_window(
    dates: Iterable[dt.date],
    *,
    timeout: float,
    retries: int,
    workers: int,
    page_size: int,
    raw_dir: Path,
    refresh: bool,
) -> dict[str, list[dict[str, Any]]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _fetch_date,
                date,
                timeout=timeout,
                retries=retries,
                page_size=page_size,
                raw_dir=raw_dir,
                refresh=refresh,
            ): date
            for date in dates
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            date_text, papers = future.result()
            results[date_text] = papers
            print(f"[{index:>3}/{len(futures)}] {date_text}: {len(papers):>2} papers")
    return dict(sorted(results.items()))


def _author_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for author in value:
        if isinstance(author, str):
            name = author
        elif isinstance(author, dict):
            name = author.get("name") or (author.get("user") or {}).get("fullname")
        else:
            name = None
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def normalize_paper(
    date_text: str,
    rank: int,
    item: dict[str, Any],
    *,
    dataset_split: str | None = None,
) -> dict[str, Any]:
    paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
    arxiv_id = paper.get("id")
    if not isinstance(arxiv_id, str) or not arxiv_id.strip():
        arxiv_id = None
    title = paper.get("title") or item.get("title")
    abstract = paper.get("summary") or item.get("summary")
    record = {
        "daily_paper_date": date_text,
        "daily_rank": rank,
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": _author_names(paper.get("authors")),
        "abstract": abstract,
        "upvotes_at_extraction": paper.get("upvotes", item.get("upvotes")),
        "hf_published_at": paper.get("publishedAt") or item.get("publishedAt"),
        "ai_summary": paper.get("ai_summary"),
        "ai_keywords": paper.get("ai_keywords"),
        "project_url": paper.get("projectPage"),
        "code_url": paper.get("githubRepo"),
        "huggingface_url": (
            f"https://huggingface.co/papers/{arxiv_id}" if arxiv_id else None
        ),
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
    }
    if dataset_split is not None:
        record["dataset_split"] = dataset_split
    return record


def normalize_window(
    responses: dict[str, list[dict[str, Any]]],
    *,
    top_k: int | None,
    dataset_split: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for date_text, papers in sorted(responses.items()):
        selected = papers if top_k is None else papers[:top_k]
        records.extend(
            normalize_paper(
                date_text,
                rank,
                item,
                dataset_split=dataset_split,
            )
            for rank, item in enumerate(selected, 1)
        )
    return records


def unique_arxiv_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        arxiv_id = record.get("arxiv_id")
        if isinstance(arxiv_id, str):
            unique.setdefault(arxiv_id, record)
    return [unique[key] for key in sorted(unique)]


def estimate_costs(
    paper_count: int,
    *,
    chars_per_token: float,
    prompt_overhead_tokens: int,
    scenarios: Iterable[CostScenario] = DEFAULT_SCENARIOS,
    prices: dict[str, PriceTier] = PRICES,
) -> dict[str, Any]:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        source_tokens = round(scenario.source_chars_per_paper / chars_per_token)
        input_tokens = source_tokens + prompt_overhead_tokens
        billed_output_tokens = (
            scenario.reasoning_tokens_per_paper
            + scenario.visible_output_tokens_per_paper
        )
        if input_tokens > SHORT_CONTEXT_LIMIT:
            raise ValueError(
                f"scenario {scenario.name} exceeds short-context pricing threshold"
            )
        tier_costs: dict[str, Any] = {}
        for tier_name, price in prices.items():
            input_cost = input_tokens / 1_000_000 * price.input_per_million
            output_cost = billed_output_tokens / 1_000_000 * price.output_per_million
            per_paper = input_cost + output_cost
            tier_costs[tier_name] = {
                "input_cost_per_paper_usd": round(input_cost, 6),
                "output_cost_per_paper_usd": round(output_cost, 6),
                "total_cost_per_paper_usd": round(per_paper, 6),
                "total_cost_all_papers_usd": round(per_paper * paper_count, 2),
            }
        rows.append(
            {
                "scenario": scenario.name,
                "source_chars_per_paper": scenario.source_chars_per_paper,
                "source_tokens_per_paper_estimate": source_tokens,
                "prompt_and_schema_tokens_per_paper": prompt_overhead_tokens,
                "input_tokens_per_paper": input_tokens,
                "reasoning_tokens_per_paper": scenario.reasoning_tokens_per_paper,
                "visible_output_tokens_per_paper": (
                    scenario.visible_output_tokens_per_paper
                ),
                "billed_output_tokens_per_paper": billed_output_tokens,
                "tiers": tier_costs,
            }
        )
    return {
        "model": MODEL,
        "pricing_as_of": PRICING_AS_OF,
        "pricing_source": OPENAI_PRICING_URL,
        "paper_count": paper_count,
        "context_price_band": "short",
        "short_context_input_threshold_tokens": SHORT_CONTEXT_LIMIT,
        "reasoning_tokens_billed_as_output": True,
        "source_char_cap_matches": "tools/target_packs/paper_summary.py",
        "max_source_chars": DEFAULT_MAX_SOURCE_CHARS,
        "chars_per_token_assumption": chars_per_token,
        "uncached_prompt_assumption": True,
        "prices_per_million_tokens": {
            name: asdict(price) for name, price in prices.items()
        },
        "scenarios": rows,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write_text(path, text)


def write_outputs(
    out_dir: Path,
    responses: dict[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
    *,
    started_at: str,
    end_date: dt.date,
    days: int,
    chars_per_token: float,
    prompt_overhead_tokens: int,
    top_k: int | None,
    dataset_split: str | None,
) -> dict[str, Any]:
    unique_records = unique_arxiv_records(records)
    dates = date_window(end_date, days)
    date_start = dates[0].isoformat()
    date_end = dates[-1].isoformat()
    nonempty_dates = sum(bool(papers) for papers in responses.values())
    empty_dates = [date for date, papers in responses.items() if not papers]
    available_occurrences = sum(len(papers) for papers in responses.values())

    _write_jsonl(out_dir / "daily_papers.jsonl", records)
    _atomic_write_text(
        out_dir / "arxiv_ids.txt",
        "".join(f"{record['arxiv_id']}\n" for record in unique_records),
    )

    counts_path = out_dir / "daily_counts.csv"
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    counts_tmp = counts_path.with_suffix(".csv.tmp")
    with counts_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "date",
                "available_paper_count",
                "selected_paper_count",
            ),
        )
        writer.writeheader()
        for date_text, papers in responses.items():
            writer.writerow(
                {
                    "date": date_text,
                    "available_paper_count": len(papers),
                    "selected_paper_count": (
                        len(papers) if top_k is None else min(len(papers), top_k)
                    ),
                }
            )
    counts_tmp.replace(counts_path)

    cost_estimate = estimate_costs(
        len(unique_records),
        chars_per_token=chars_per_token,
        prompt_overhead_tokens=prompt_overhead_tokens,
    )
    _atomic_write_text(
        out_dir / "cost_estimate.json",
        json.dumps(cost_estimate, ensure_ascii=False, indent=2) + "\n",
    )

    manifest = {
        "source": "Hugging Face Daily Papers public API",
        "api_template": HF_API_TEMPLATE,
        "extraction_started_at": started_at,
        "extraction_finished_at": dt.datetime.now(dt.UTC).isoformat(),
        "date_start": date_start,
        "date_end": date_end,
        "calendar_days": days,
        "top_k_per_date": top_k,
        "dataset_split": dataset_split,
        "nonempty_dates": nonempty_dates,
        "empty_dates": empty_dates,
        "available_paper_occurrences": available_occurrences,
        "selected_paper_occurrences": len(records),
        "unique_arxiv_papers": len(unique_records),
        "duplicate_occurrences": len(records) - len(unique_records),
        "outputs": {
            "normalized_records": "daily_papers.jsonl",
            "daily_counts": "daily_counts.csv",
            "unique_arxiv_ids": "arxiv_ids.txt",
            "cost_estimate": "cost_estimate.json",
            "raw_responses": "raw/YYYY-MM-DD.json",
        },
    }
    _atomic_write_text(
        out_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def _load_existing(out_dir: Path) -> tuple[dict[str, Any], int]:
    manifest_path = out_dir / "manifest.json"
    ids_path = out_dir / "arxiv_ids.txt"
    if not manifest_path.is_file() or not ids_path.is_file():
        raise FileNotFoundError(
            "--estimate-only requires manifest.json and arxiv_ids.txt in --out-dir"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paper_count = sum(1 for line in ids_path.read_text(encoding="utf-8").splitlines() if line)
    return manifest, paper_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument(
        "--end-date",
        type=dt.date.fromisoformat,
        default=dt.date.today(),
        help="inclusive YYYY-MM-DD end date; default: local today",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/hf-daily-papers"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="retain this many highest-ranked papers per date; use 0 to retain all",
    )
    parser.add_argument(
        "--split-name",
        help="optional dataset split label added to every normalized record and manifest",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="Hugging Face page size; the public API currently accepts at most 100",
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore cached raw/YYYY-MM-DD.json responses",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="recompute cost_estimate.json from an existing extraction",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=DEFAULT_CHARS_PER_TOKEN,
    )
    parser.add_argument(
        "--prompt-overhead-tokens",
        type=int,
        default=DEFAULT_PROMPT_OVERHEAD_TOKENS,
        help="developer prompt plus JSON schema token allowance per paper",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")
    if not 1 <= args.page_size <= 100:
        raise SystemExit("--page-size must be between 1 and 100")
    if args.top_k < 0:
        raise SystemExit("--top-k cannot be negative")
    top_k = args.top_k or None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.estimate_only:
        manifest, paper_count = _load_existing(args.out_dir)
        estimate = estimate_costs(
            paper_count,
            chars_per_token=args.chars_per_token,
            prompt_overhead_tokens=args.prompt_overhead_tokens,
        )
        _atomic_write_text(
            args.out_dir / "cost_estimate.json",
            json.dumps(estimate, ensure_ascii=False, indent=2) + "\n",
        )
        print(
            f"Re-estimated {paper_count} unique papers from "
            f"{manifest['date_start']} through {manifest['date_end']}."
        )
        return 0

    dates = date_window(args.end_date, args.days)
    started_at = dt.datetime.now(dt.UTC).isoformat()
    print(
        f"Fetching {len(dates)} calendar days: "
        f"{dates[0].isoformat()} through {dates[-1].isoformat()}"
    )
    responses = fetch_window(
        dates,
        timeout=args.timeout,
        retries=args.retries,
        workers=args.workers,
        page_size=args.page_size,
        raw_dir=args.out_dir / "raw",
        refresh=args.refresh,
    )
    records = normalize_window(
        responses,
        top_k=top_k,
        dataset_split=args.split_name,
    )
    manifest = write_outputs(
        args.out_dir,
        responses,
        records,
        started_at=started_at,
        end_date=args.end_date,
        days=args.days,
        chars_per_token=args.chars_per_token,
        prompt_overhead_tokens=args.prompt_overhead_tokens,
        top_k=top_k,
        dataset_split=args.split_name,
    )
    print(
        f"DONE: retained {manifest['selected_paper_occurrences']} of "
        f"{manifest['available_paper_occurrences']} occurrences, "
        f"{manifest['unique_arxiv_papers']} unique papers, "
        f"{manifest['nonempty_dates']}/{manifest['calendar_days']} non-empty dates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
