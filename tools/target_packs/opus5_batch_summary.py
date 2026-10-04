"""Generate paper golden summaries with Claude Opus 5 Message Batches.

The runner is resumable. It downloads and flattens the same arXiv LaTeX used by
``paper_summary.py``, submits only missing papers, polls the batch, validates the
structured JSON, and records exact API usage plus usage-derived list-price cost.

Example:

    python tools/target_packs/opus5_batch_summary.py \
      --papers artifacts/my-validation-set/daily_papers.jsonl \
      --out-dir artifacts/my-validation-set/opus5_summaries

``ANTHROPIC_API_KEY`` must already be present in the environment. Message Batch
requests can take up to 24 hours; rerunning this command resumes an active batch.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any

from tech_tree_arena.runtime.provider_client import _validate_json_schema
try:
    from tools.target_packs.paper_summary import (
        DEVELOPER_PROMPT,
        SUMMARY_SCHEMA,
        build_user_prompt,
        download_source,
        extract_source_text,
        render_markdown,
    )
except ModuleNotFoundError:  # direct ``python path/to/script.py`` execution
    from paper_summary import (  # type: ignore[no-redef]
        DEVELOPER_PROMPT,
        SUMMARY_SCHEMA,
        build_user_prompt,
        download_source,
        extract_source_text,
        render_markdown,
    )


DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 16_000
PRICING_AS_OF = "2026-08-16"
PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"

# Global Claude API Message Batch list prices, USD per million tokens.
BATCH_PRICES = {
    "input": Decimal("2.50"),
    "cache_write_5m": Decimal("3.125"),
    "cache_write_1h": Decimal("5.00"),
    "cache_read": Decimal("0.25"),
    "output": Decimal("12.50"),
}
STANDARD_PRICES = {
    "input": Decimal("5.00"),
    "cache_write_5m": Decimal("6.25"),
    "cache_write_1h": Decimal("10.00"),
    "cache_read": Decimal("0.50"),
    "output": Decimal("25.00"),
}


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


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


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _load_papers(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    papers: list[dict[str, Any]] = []
    for row in rows:
        arxiv_id = row.get("arxiv_id")
        if not isinstance(arxiv_id, str) or not arxiv_id:
            raise ValueError("every paper must have a non-empty arxiv_id")
        if arxiv_id in seen:
            raise ValueError(f"duplicate arXiv id in input: {arxiv_id}")
        seen.add(arxiv_id)
        papers.append(row)
    if not papers:
        raise ValueError("paper input is empty")
    return papers


def _custom_id(arxiv_id: str) -> str:
    return "arxiv-" + arxiv_id.replace(".", "-").replace("/", "-")


def _cost_from_usage(
    usage: dict[str, Any],
    prices: dict[str, Decimal] = BATCH_PRICES,
) -> dict[str, Any]:
    cache_creation = usage.get("cache_creation") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cache_read_tokens = int(usage.get("cache_read_input_tokens") or 0)
    cache_write_5m_tokens = int(
        cache_creation.get("ephemeral_5m_input_tokens") or 0
    )
    cache_write_1h_tokens = int(
        cache_creation.get("ephemeral_1h_input_tokens") or 0
    )
    declared_cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    detailed_cache_creation = cache_write_5m_tokens + cache_write_1h_tokens
    if declared_cache_creation and not detailed_cache_creation:
        # The SDK/API historically exposed only a total. This run requests the
        # default 5-minute cache, so attribute an undifferentiated total to 5m.
        cache_write_5m_tokens = declared_cache_creation

    line_items = {
        "input": Decimal(input_tokens) * prices["input"] / 1_000_000,
        "cache_write_5m": (
            Decimal(cache_write_5m_tokens)
            * prices["cache_write_5m"]
            / 1_000_000
        ),
        "cache_write_1h": (
            Decimal(cache_write_1h_tokens)
            * prices["cache_write_1h"]
            / 1_000_000
        ),
        "cache_read": (
            Decimal(cache_read_tokens) * prices["cache_read"] / 1_000_000
        ),
        "output": Decimal(output_tokens) * prices["output"] / 1_000_000,
    }
    total = sum(line_items.values(), Decimal("0"))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_write_5m_tokens": cache_write_5m_tokens,
        "cache_write_1h_tokens": cache_write_1h_tokens,
        "cache_read_tokens": cache_read_tokens,
        "line_items_usd": {
            key: float(value.quantize(Decimal("0.00000001")))
            for key, value in line_items.items()
        },
        "total_usd": float(total.quantize(Decimal("0.00000001"))),
    }


def _pdf_source_for_paper(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    arxiv_id = row["arxiv_id"]
    pdf_url = row.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}"
    request = urllib.request.Request(
        pdf_url,
        headers={"User-Agent": "tech_tree_repro/opus5_batch_summary"},
    )
    with urllib.request.urlopen(request, timeout=120.0) as response:
        pdf_bytes = response.read()
    if not pdf_bytes.startswith(b"%PDF"):
        raise RuntimeError(f"{arxiv_id}: PDF endpoint did not return a PDF")
    with tempfile.TemporaryDirectory(prefix="idea-arena-paper-") as temporary:
        pdf_path = Path(temporary) / f"{arxiv_id}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            timeout=180.0,
        )
    text = completed.stdout.decode("utf-8", errors="ignore").strip()
    if not text:
        raise RuntimeError(f"{arxiv_id}: PDF text extraction was empty")
    if len(text) > 140_000:
        text = text[:140_000]
    return text, {
        "source_kind": "arxiv_pdf_text",
        "source_chars": len(text),
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_archive": None,
        "source_pdf_url": pdf_url,
    }


def _abstract_source_for_paper(
    row: dict[str, Any], source_errors: list[str]
) -> tuple[str, dict[str, Any]]:
    arxiv_id = row["arxiv_id"]
    abstract = str(row.get("abstract") or "").strip()
    if not abstract:
        raise RuntimeError(f"{arxiv_id}: no LaTeX, PDF text, or abstract available")
    authors = ", ".join(str(author) for author in row.get("authors") or [])
    text = "\n\n".join(
        part
        for part in (
            f"Title: {row.get('title') or arxiv_id}",
            f"Authors: {authors}" if authors else "",
            f"Abstract:\n{abstract}",
        )
        if part
    )
    return text, {
        "source_kind": "metadata_abstract_fallback",
        "source_chars": len(text),
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_archive": None,
        "source_errors": source_errors,
    }


def _source_for_paper(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    arxiv_id = row["arxiv_id"]
    source_errors: list[str] = []
    for attempt in range(1, 4):
        try:
            archive = download_source(arxiv_id)
            text = extract_source_text(archive, arxiv_id)
            return text, {
                "source_kind": "arxiv_latex",
                "source_chars": len(text),
                "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "source_archive": str(archive),
            }
        except Exception as exc:  # source retrieval/extraction fallback boundary
            source_errors.append(
                f"latex attempt {attempt}: {type(exc).__name__}: {str(exc)[:500]}"
            )
            if "HTTP Error 404" in str(exc):
                break
            if attempt < 3:
                time.sleep(2**attempt)
    try:
        text, metadata = _pdf_source_for_paper(row)
        metadata["source_errors"] = source_errors
        print(f"  [fallback] {arxiv_id}: using extracted PDF text")
        return text, metadata
    except Exception as exc:
        source_errors.append(f"pdf: {type(exc).__name__}: {str(exc)[:500]}")
    print(f"  [fallback] {arxiv_id}: using local metadata abstract")
    return _abstract_source_for_paper(row, source_errors)


def _request_for_paper(
    row: dict[str, Any],
    source_text: str,
    *,
    model: str,
    effort: str,
    max_tokens: int,
    enable_prompt_cache: bool,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "inference_geo": "global",
        "system": DEVELOPER_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": build_user_prompt(row["arxiv_id"], source_text),
            }
        ],
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": SUMMARY_SCHEMA},
        },
    }
    if enable_prompt_cache:
        params["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
    return {
        "custom_id": _custom_id(row["arxiv_id"]),
        "params": params,
    }


def _text_from_message(message: Any) -> str:
    parts = [
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    if not parts:
        raise ValueError("successful message contains no text block")
    return "".join(parts)


def _normalize_summary(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Enforce prompt-level cardinality not expressible in provider JSON schema."""
    findings = summary.get("key_findings")
    if not isinstance(findings, list) or len(findings) <= 5:
        return summary, None
    normalized = dict(summary)
    normalized["key_findings"] = findings[:5]
    return normalized, {
        "operation": "truncate_key_findings_to_prompt_limit",
        "original_count": len(findings),
        "retained_count": 5,
        "rule": "The prompt orders findings most-to-least important and requests 3-5.",
    }


def _write_outputs(
    *,
    out_dir: Path,
    paper: dict[str, Any],
    source_metadata: dict[str, Any],
    batch_id: str,
    message: Any,
    model: str,
    effort: str,
    service_tier: str,
    prices: dict[str, Decimal],
) -> dict[str, Any]:
    arxiv_id = paper["arxiv_id"]
    raw_summary = json.loads(_text_from_message(message))
    _validate_json_schema(raw_summary, SUMMARY_SCHEMA)
    summary, postprocessing = _normalize_summary(raw_summary)
    usage = _model_dump(message.usage)
    cost = _cost_from_usage(usage, prices)
    record = {
        "arxiv_id": arxiv_id,
        "daily_paper_date": paper.get("daily_paper_date"),
        "daily_rank": paper.get("daily_rank"),
        "dataset_split": paper.get("dataset_split"),
        "paper_title": paper.get("title"),
        "model": model,
        "effort": effort,
        "service_tier": service_tier,
        "inference_geo": "global",
        "batch_id": batch_id,
        "message_id": message.id,
        "stop_reason": message.stop_reason,
        "source": source_metadata,
        "usage": usage,
        "usage_derived_cost": cost,
        "summary": summary,
    }
    if postprocessing is not None:
        record["raw_summary"] = raw_summary
        record["postprocessing"] = postprocessing
    _atomic_json(out_dir / "json" / f"{arxiv_id}.json", record)
    _atomic_text(out_dir / "markdown" / f"{arxiv_id}.md", render_markdown(arxiv_id, summary))
    return record


def _aggregate(
    out_dir: Path,
    papers: list[dict[str, Any]],
    *,
    model: str,
    effort: str,
    service_tier: str,
    prices: dict[str, Decimal],
    state: dict[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for paper in papers:
        path = out_dir / "json" / f"{paper['arxiv_id']}.json"
        if path.is_file():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    token_fields = (
        "input_tokens",
        "output_tokens",
        "cache_write_5m_tokens",
        "cache_write_1h_tokens",
        "cache_read_tokens",
    )
    totals = {
        field: sum(record["usage_derived_cost"][field] for record in records)
        for field in token_fields
    }
    total_usd = sum(
        Decimal(str(record["usage_derived_cost"]["total_usd"]))
        for record in records
    )
    thinking_tokens = sum(
        int(
            (record["usage"].get("output_tokens_details") or {}).get(
                "thinking_tokens", 0
            )
            or 0
        )
        for record in records
    )
    totals["thinking_tokens"] = thinking_tokens
    cost_report = {
        "status": "complete" if len(records) == len(papers) else "partial",
        "generated_at": _utc_now(),
        "model": model,
        "effort": effort,
        "service_tier": service_tier,
        "inference_geo": "global",
        "paper_count_requested": len(papers),
        "paper_count_succeeded": len(records),
        "paper_count_missing": len(papers) - len(records),
        "batch_ids": [
            round_["batch_id"]
            for round_ in state.get("rounds", [])
            if round_.get("batch_id")
        ],
        "usage_totals": totals,
        "actual_list_price_cost_usd": float(total_usd.quantize(Decimal("0.000001"))),
        "pricing_as_of": PRICING_AS_OF,
        "pricing_source": PRICING_SOURCE,
        "prices_per_million_tokens_usd": {
            key: float(value) for key, value in prices.items()
        },
        "cost_scope": (
            "Usage-derived Anthropic list-price model cost. It excludes taxes and "
            "reflects neither account credits nor negotiated discounts."
        ),
    }
    _atomic_json(out_dir / "actual_cost.json", cost_report)
    _atomic_text(
        out_dir / "golden_summaries.jsonl",
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )
    cost_csv = out_dir / "per_paper_cost.csv"
    cost_csv.parent.mkdir(parents=True, exist_ok=True)
    cost_csv_tmp = cost_csv.with_suffix(".csv.tmp")
    with cost_csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "daily_paper_date",
            "arxiv_id",
            "paper_title",
            "source_chars",
            "input_tokens",
            "cache_write_5m_tokens",
            "cache_read_tokens",
            "output_tokens",
            "thinking_tokens",
            "cost_usd",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            usage = record["usage"]
            billed = record["usage_derived_cost"]
            writer.writerow(
                {
                    "daily_paper_date": record.get("daily_paper_date"),
                    "arxiv_id": record["arxiv_id"],
                    "paper_title": record.get("paper_title"),
                    "source_chars": record["source"]["source_chars"],
                    "input_tokens": billed["input_tokens"],
                    "cache_write_5m_tokens": billed["cache_write_5m_tokens"],
                    "cache_read_tokens": billed["cache_read_tokens"],
                    "output_tokens": billed["output_tokens"],
                    "thinking_tokens": int(
                        (usage.get("output_tokens_details") or {}).get(
                            "thinking_tokens", 0
                        )
                        or 0
                    ),
                    "cost_usd": f"{billed['total_usd']:.8f}",
                }
            )
    cost_csv_tmp.replace(cost_csv)
    return cost_report


def _load_state(path: Path, *, model: str, effort: str) -> dict[str, Any]:
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("model") != model or state.get("effort") != effort:
            raise ValueError("existing batch state uses a different model or effort")
        return state
    return {
        "version": 1,
        "created_at": _utc_now(),
        "model": model,
        "effort": effort,
        "rounds": [],
    }


def _run_standard(
    args: argparse.Namespace,
    papers: list[dict[str, Any]],
    state: dict[str, Any],
) -> int:
    from anthropic import Anthropic

    source_payloads: dict[str, tuple[str, dict[str, Any]]] = {}
    missing = [
        paper
        for paper in papers
        if not (args.out_dir / "json" / f"{paper['arxiv_id']}.json").is_file()
    ]
    print(f"Preparing {len(missing)} paper source(s) for standard requests...")
    for index, paper in enumerate(missing, 1):
        print(f"[{index:>2}/{len(missing)}] {paper['arxiv_id']}")
        source_payloads[paper["arxiv_id"]] = _source_for_paper(paper)

    def call_one(paper: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        source_text, _ = source_payloads[paper["arxiv_id"]]
        request = _request_for_paper(
            paper,
            source_text,
            model=args.model,
            effort=args.effort,
            max_tokens=args.max_tokens,
            enable_prompt_cache=args.enable_prompt_cache,
        )["params"]
        client = Anthropic(max_retries=4, timeout=600.0)
        return paper, client.messages.create(**request)

    failures: list[dict[str, Any]] = []
    for attempt in range(1, args.max_rounds + 1):
        pending = [
            paper
            for paper in missing
            if not (args.out_dir / "json" / f"{paper['arxiv_id']}.json").is_file()
        ]
        if not pending:
            break
        print(
            f"Standard round {attempt}: submitting {len(pending)} request(s) "
            f"with concurrency={args.concurrency}"
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            futures = {executor.submit(call_one, paper): paper for paper in pending}
            for future in concurrent.futures.as_completed(futures):
                paper = futures[future]
                arxiv_id = paper["arxiv_id"]
                try:
                    _, message = future.result()
                    _, metadata = source_payloads[arxiv_id]
                    record = _write_outputs(
                        out_dir=args.out_dir,
                        paper=paper,
                        source_metadata=metadata,
                        batch_id="standard-direct",
                        message=message,
                        model=args.model,
                        effort=args.effort,
                        service_tier="standard",
                        prices=STANDARD_PRICES,
                    )
                    print(
                        f"Wrote {arxiv_id}: input="
                        f"{record['usage'].get('input_tokens', 0)} output="
                        f"{record['usage'].get('output_tokens', 0)} cost="
                        f"${record['usage_derived_cost']['total_usd']:.4f}"
                    )
                except Exception as exc:  # continue and retry only missing papers
                    error = {
                        "attempt": attempt,
                        "arxiv_id": arxiv_id,
                        "type": type(exc).__name__,
                        "message": str(exc)[:1000],
                        "recorded_at": _utc_now(),
                    }
                    failures.append(error)
                    print(f"ERROR {arxiv_id}: {error['type']}: {error['message']}")
        state.setdefault("rounds", []).append(
            {
                "round": attempt,
                "service_tier": "standard",
                "completed_at": _utc_now(),
                "failures": [error for error in failures if error["attempt"] == attempt],
            }
        )
        _atomic_json(args.out_dir / "standard_state.json", state)

    report = _aggregate(
        args.out_dir,
        papers,
        model=args.model,
        effort=args.effort,
        service_tier="standard",
        prices=STANDARD_PRICES,
        state=state,
    )
    print(
        f"DONE: {report['paper_count_succeeded']}/{report['paper_count_requested']} "
        f"summaries; usage-derived list-price cost="
        f"${report['actual_list_price_cost_usd']:.6f}"
    )
    return 0 if report["status"] == "complete" else 1


def run(args: argparse.Namespace) -> int:
    from anthropic import Anthropic

    papers = _load_papers(args.papers)
    by_custom_id = {_custom_id(row["arxiv_id"]): row for row in papers}
    source_metadata: dict[str, dict[str, Any]] = {}
    state_path = args.out_dir / (
        "batch_state.json" if args.service_tier == "batch" else "standard_state.json"
    )
    state = _load_state(state_path, model=args.model, effort=args.effort)
    if args.service_tier == "standard":
        return _run_standard(args, papers, state)
    client = Anthropic(max_retries=2, timeout=120.0)

    for round_number in range(1, args.max_rounds + 1):
        missing = [
            paper
            for paper in papers
            if not (args.out_dir / "json" / f"{paper['arxiv_id']}.json").is_file()
        ]
        if not missing:
            break

        active = next(
            (
                round_
                for round_ in reversed(state["rounds"])
                if round_.get("processing_status") != "ended"
            ),
            None,
        )
        if active is None:
            requests: list[dict[str, Any]] = []
            print(f"Preparing {len(missing)} paper source(s) for batch round {round_number}...")
            for index, paper in enumerate(missing, 1):
                arxiv_id = paper["arxiv_id"]
                print(f"[{index:>2}/{len(missing)}] {arxiv_id}")
                source_text, metadata = _source_for_paper(paper)
                source_metadata[arxiv_id] = metadata
                requests.append(
                    _request_for_paper(
                        paper,
                        source_text,
                        model=args.model,
                        effort=args.effort,
                        max_tokens=args.max_tokens,
                        enable_prompt_cache=args.enable_prompt_cache,
                    )
                )
            print(f"Submitting {len(requests)} request(s) to {args.model} Message Batch...")
            batch = client.messages.batches.create(requests=requests)
            active = {
                "round": len(state["rounds"]) + 1,
                "batch_id": batch.id,
                "submitted_at": _utc_now(),
                "custom_ids": [request["custom_id"] for request in requests],
                "processing_status": batch.processing_status,
                "request_counts": _model_dump(batch.request_counts),
            }
            state["rounds"].append(active)
            _atomic_json(state_path, state)
            print(f"Submitted batch {batch.id}")

        batch_id = active["batch_id"]
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            active["processing_status"] = batch.processing_status
            active["request_counts"] = _model_dump(batch.request_counts)
            active["last_checked_at"] = _utc_now()
            if batch.ended_at is not None:
                active["ended_at"] = batch.ended_at.isoformat()
            _atomic_json(state_path, state)
            counts = active["request_counts"]
            print(
                f"{batch_id}: {batch.processing_status}; "
                f"processing={counts.get('processing', 0)} "
                f"succeeded={counts.get('succeeded', 0)} "
                f"errored={counts.get('errored', 0)}"
            )
            if batch.processing_status == "ended":
                break
            if args.no_wait:
                print("Batch is still running; rerun the same command to resume.")
                return 2
            time.sleep(args.poll_seconds)

        errors: list[dict[str, Any]] = []
        for entry in client.messages.batches.results(batch_id):
            custom_id = entry.custom_id
            paper = by_custom_id.get(custom_id)
            if paper is None:
                errors.append({"custom_id": custom_id, "error": "unknown custom id"})
                continue
            result = entry.result
            if result.type != "succeeded":
                errors.append(
                    {
                        "custom_id": custom_id,
                        "arxiv_id": paper["arxiv_id"],
                        "result": _model_dump(result),
                    }
                )
                continue
            arxiv_id = paper["arxiv_id"]
            metadata = source_metadata.get(arxiv_id)
            if metadata is None:
                source_text, metadata = _source_for_paper(paper)
                source_metadata[arxiv_id] = metadata
            record = _write_outputs(
                out_dir=args.out_dir,
                paper=paper,
                source_metadata=metadata,
                batch_id=batch_id,
                message=result.message,
                model=args.model,
                effort=args.effort,
                service_tier="message_batch",
                prices=BATCH_PRICES,
            )
            print(
                f"Wrote {arxiv_id}: input={record['usage'].get('input_tokens', 0)} "
                f"output={record['usage'].get('output_tokens', 0)} "
                f"cost=${record['usage_derived_cost']['total_usd']:.4f}"
            )
        active["results_consumed_at"] = _utc_now()
        active["result_errors"] = errors
        _atomic_json(state_path, state)

    report = _aggregate(
        args.out_dir,
        papers,
        model=args.model,
        effort=args.effort,
        service_tier="message_batch",
        prices=BATCH_PRICES,
        state=state,
    )
    print(
        f"DONE: {report['paper_count_succeeded']}/{report['paper_count_requested']} "
        f"summaries; usage-derived list-price cost="
        f"${report['actual_list_price_cost_usd']:.6f}"
    )
    return 0 if report["status"] == "complete" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--service-tier",
        choices=("batch", "standard"),
        default="batch",
    )
    parser.add_argument(
        "--effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default=DEFAULT_EFFORT,
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--enable-prompt-cache",
        action="store_true",
        help=(
            "enable Anthropic automatic 5-minute prompt caching; leave disabled "
            "for one-shot unique paper sources to avoid cache-write premiums"
        ),
    )
    parser.add_argument("--no-wait", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    if args.max_rounds < 1:
        raise SystemExit("--max-rounds must be positive")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
