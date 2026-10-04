"""Self-contained arXiv paper summarizer in the Standard Paper Summary Format.

Given one or more arXiv ids, this script:
  1. Downloads the paper's LaTeX source archive from arXiv (per-paper e-print
     endpoint, with an export-mirror fallback), caching it on disk. If a local
     bulk tar cache for the paper's month already exists it is used first.
  2. Extracts the archive, picks the LaTeX entrypoint, inlines \\input/\\include,
     and strips comments to recover a clean flattened source body.
  3. Asks an LLM to distil the paper into the Standard Paper Summary Format
     (1. Setting and main object [merged, self-contained crux] / 2. Concrete
     detailed setting [hierarchical] / 3. Novel key findings) via a strict JSON
     schema.
  4. Writes a JSON record and a rendered Markdown summary per paper.

It uses the Arena-owned provider client. The arXiv fetch + LaTeX extraction logic is
inlined here (adapted from stages/shared/tex.py) so the script does not pull in
the heavyweight legacy source-content pipeline.

Usage:
  python tools/target_packs/paper_summary.py 2509.01229 2509.11106 ...
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from tech_tree_arena.resources import arena_home
from tech_tree_arena.runtime import provider_client

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

SUMMARY_MODEL = "gpt-5.5"
MAX_SOURCE_CHARS = 140_000          # cap flattened LaTeX fed to the model
MAX_TEX_FILES = 200                 # recursion guard for \input flattening
HTTP_TIMEOUT = 90.0
CACHE_DIR = Path(os.environ.get("IDEA_ARENA_PAPER_CACHE", arena_home() / "cache" / "paper_src"))
DEFAULT_OUT_DIR = Path(os.environ.get("IDEA_ARENA_SUMMARY_DIR", arena_home() / "summaries"))
BULK_ROOT = Path(os.environ.get("IDEA_ARENA_ARXIV_BULK_ROOT", arena_home() / "arxiv_source_bulk"))

# --------------------------------------------------------------------------- #
# 1. Fetch arXiv source
# --------------------------------------------------------------------------- #


def _source_url_candidates(arxiv_id: str) -> list[str]:
    return [
        f"https://arxiv.org/src/{arxiv_id}",
        f"https://export.arxiv.org/src/{arxiv_id}",
        f"https://arxiv.org/e-print/{arxiv_id}",
        f"https://export.arxiv.org/e-print/{arxiv_id}",
    ]


def _try_local_bulk(arxiv_id: str, dest: Path, log) -> bool:
    """Extract this paper's .gz source from a locally-cached monthly bulk tar.

    Mirrors the Kaggle/S3 bulk layout: artifacts/arxiv_source_bulk/<yymm>/tars/*.tar
    each containing <yymm>/<arxiv_id>.gz entries. Returns True if found.
    """
    yymm = arxiv_id.split(".")[0]
    tar_dir = BULK_ROOT / yymm / "tars"
    if not tar_dir.is_dir():
        return False
    want = f"{arxiv_id}.gz"
    for tar_path in sorted(tar_dir.glob("*.tar")):
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                member = next(
                    (m for m in tar.getmembers()
                     if m.name.endswith(want) and m.isfile()), None)
                if member is None:
                    continue
                fobj = tar.extractfile(member)
                if fobj is None:
                    continue
                dest.write_bytes(fobj.read())
                log(f"  [bulk] extracted {arxiv_id} from {tar_path.name}")
                return True
        except (tarfile.TarError, OSError):
            continue
    return False


def download_source(arxiv_id: str, log=print) -> Path:
    """Return a path to the cached source archive for ``arxiv_id`` (download if needed)."""
    if not re.fullmatch(r"[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?", arxiv_id):
        raise ValueError("expected a modern arXiv ID")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive = CACHE_DIR / f"{arxiv_id}.tar.gz"
    if archive.exists() and archive.stat().st_size > 0:
        log(f"  [cache] {archive}")
        return archive

    if _try_local_bulk(arxiv_id, archive, log) and archive.stat().st_size > 0:
        return archive

    last_err: Exception | None = None
    for url in _source_url_candidates(arxiv_id):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "tech_tree_repro/paper_summary"})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                data = resp.read()
            if not data:
                raise RuntimeError("empty download")
            archive.write_bytes(data)
            log(f"  [http] {url} -> {len(data):,} bytes")
            return archive
        except Exception as exc:  # noqa: BLE001 - try next mirror
            last_err = exc
            log(f"  [http] {url} failed: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"could not fetch source for {arxiv_id}: {last_err}")


# --------------------------------------------------------------------------- #
# 2. Extract + flatten LaTeX  (inlined from stages/shared/tex.py)
# --------------------------------------------------------------------------- #


def strip_comments(text: str) -> str:
    """Remove LaTeX ``%`` comments while preserving escaped percent signs."""
    lines: list[str] = []
    for line in text.splitlines():
        out: list[str] = []
        escaped = False
        for char in line:
            if char == "%" and not escaped:
                break
            out.append(char)
            escaped = char == "\\"
        lines.append("".join(out))
    return "\n".join(lines)


def _read_tex(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _resolve_tex_path(base_dir: Path, ref: str) -> Path | None:
    candidate = ref.strip()
    if not candidate or candidate.endswith(
            (".png", ".jpg", ".jpeg", ".pdf", ".eps", ".sty", ".cls")):
        return None
    path = (base_dir / candidate).resolve()
    options = [path]
    if path.suffix == "":
        options.append(path.with_suffix(".tex"))
    elif path.suffix.lower() != ".tex":
        options.append(path.with_name(f"{path.name}.tex"))
    for item in options:
        if item.is_file():
            return item
    return None


def _flatten_tex(path: Path, visited: set[Path], read_files: list[Path], root: Path | None = None) -> str:
    root = root or path.parent.resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError("LaTeX include escapes the source directory")
    if path in visited:
        return ""
    visited.add(path)
    read_files.append(path)
    raw = strip_comments(_read_tex(path))
    pattern = re.compile(r"\\(?:input|include)\{([^}]+)\}")
    parts: list[str] = []
    cursor = 0
    for match in pattern.finditer(raw):
        parts.append(raw[cursor:match.start()])
        inc = _resolve_tex_path(path.parent, match.group(1))
        if inc is not None and len(read_files) < MAX_TEX_FILES:
            parts.append(_flatten_tex(inc, visited, read_files, root))
        cursor = match.end()
    parts.append(raw[cursor:])
    return "".join(parts)


def _find_entrypoint(root: Path) -> Path | None:
    """Pick the most plausible top-level .tex (\\documentclass + \\begin{document})."""
    candidates: list[tuple[int, int, int, Path]] = []
    for tex in root.rglob("*.tex"):
        text = _read_tex(tex)
        if "\\documentclass" not in text or "\\begin{document}" not in text:
            continue
        root_penalty = 0 if tex.parent == root else 1
        include_score = len(re.findall(r"\\(?:input|include)\{", text))
        candidates.append((root_penalty, -include_score, -tex.stat().st_size, tex))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][-1]


def extract_source_text(archive: Path, arxiv_id: str, log=print) -> str:
    """Extract archive and return flattened, comment-stripped LaTeX body text."""
    extract_dir = CACHE_DIR / arxiv_id
    extract_dir.mkdir(parents=True, exist_ok=True)

    # The e-print blob is usually a gzipped tar, but may be a single gzipped
    # .tex (older single-file submissions) or even a bare PDF.
    extracted_any = False
    try:
        with tarfile.open(archive, "r:*") as tar:
            for member in tar.getmembers():
                destination = (extract_dir / member.name).resolve()
                if not destination.is_relative_to(extract_dir.resolve()):
                    raise ValueError("archive member escapes the source directory")
                if not (member.isfile() or member.isdir()):
                    raise ValueError("source archive contains a link or special file")
            tar.extractall(extract_dir, filter="data")
        extracted_any = any(extract_dir.rglob("*.tex"))
    except tarfile.TarError:
        pass

    if not extracted_any:
        # Try treating the archive as a single gzipped file.
        try:
            with gzip.open(archive, "rb") as fh:
                blob = fh.read()
        except OSError:
            blob = archive.read_bytes()
        if blob[:4] == b"%PDF":
            raise RuntimeError(f"{arxiv_id}: source is PDF-only (no LaTeX)")
        single = extract_dir / "main.tex"
        single.write_text(blob.decode("utf-8", errors="ignore"))

    entry = _find_entrypoint(extract_dir)
    if entry is None:
        # No documentclass found: concatenate every .tex by size as a fallback.
        texs = sorted(extract_dir.rglob("*.tex"),
                      key=lambda p: p.stat().st_size, reverse=True)
        if not texs:
            raise RuntimeError(f"{arxiv_id}: no .tex files in source")
        log(f"  [tex] no entrypoint; concatenating {len(texs)} .tex files")
        body = "\n\n".join(strip_comments(_read_tex(t)) for t in texs)
    else:
        read_files: list[Path] = []
        body = _flatten_tex(entry, set(), read_files, extract_dir.resolve())
        log(f"  [tex] entrypoint {entry.name}; flattened {len(read_files)} file(s)")

    body = body.strip()
    if len(body) > MAX_SOURCE_CHARS:
        log(f"  [tex] truncating {len(body):,} -> {MAX_SOURCE_CHARS:,} chars")
        body = body[:MAX_SOURCE_CHARS]
    return body


# --------------------------------------------------------------------------- #
# 3. Summarize via LLM (Standard Paper Summary Format)
# --------------------------------------------------------------------------- #

SUMMARY_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string", "description": "The paper's title."},
        "setting_and_object": {
            "type": "object",
            "additionalProperties": False,
            "description": "The paper's single central object and the core problem "
                           "it addresses -- kept CRISP. The main object is the one "
                           "thing the paper fundamentally contributes or studies "
                           "(for an algorithm paper, the algorithm itself and its "
                           "defining mechanism; for theory, the central result; for "
                           "a benchmark, the artifact; for a software/system paper, "
                           "the artifact and its key design principle(s)). After "
                           "reading this a reader should grasp 'what is this paper, "
                           "at its core?'.",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The main object's category, e.g. Algorithm / "
                                   "Theory / Benchmark / Architecture / Dataset / "
                                   "Mechanism / Scaling law / Phenomenon / "
                                   "Evaluation protocol.",
                },
                "groups": {
                    "type": "array",
                    "description": "Grouped aspects covering ONLY the core problem/"
                                   "motivation and the main object + crux of the "
                                   "approach. The task/domain, data, regime, "
                                   "objective/metric, and all evaluation context "
                                   "belong in the concrete detailed setting, not "
                                   "here. Very tight: 1-2 groups, each with only "
                                   "the 1-3 most essential items.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "aspect": {
                                "type": "string",
                                "description": "Short heading, e.g. 'Problem & "
                                               "motivation', 'Setting', 'Main "
                                               "object', 'Core idea / crux', "
                                               "'Objective'.",
                            },
                            "details": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Detail items under this aspect.",
                            },
                        },
                        "required": ["aspect", "details"],
                    },
                },
            },
            "required": ["category", "groups"],
        },
        "concrete_detailed_setting": {
            "type": "array",
            "description": "The concrete choices the researchers decided on "
                           "*later* to instantiate and evaluate the idea, grouped "
                           "by aspect. Each group has an aspect heading and a list "
                           "of concrete detail items. Cover the relevant aspects: "
                           "exact datasets, model sizes/architectures, training & "
                           "eval regimes/budgets, key hyperparameters, baselines "
                           "compared against, ablations, and (for theory) the "
                           "formal model, assumptions, and theorem/proof setup. "
                           "All secondary specifics live here, not in the main "
                           "setting. Stay selective: 3-5 groups, only what "
                           "materially matters; do not exhaustively enumerate.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "aspect": {
                        "type": "string",
                        "description": "Short heading, e.g. 'Datasets', 'Models', "
                                       "'Training regime', 'Baselines', "
                                       "'Ablations', 'Formal model & assumptions'.",
                    },
                    "details": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Concrete detail items under this aspect.",
                    },
                },
                "required": ["aspect", "details"],
            },
        },
        "key_findings": {
            "type": "array",
            "description": "Most important first; each genuinely novel to this "
                           "paper.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "finding": {
                        "type": "string",
                        "description": "ONE concise sentence highlighting the "
                                       "finding/takeaway. Plain prose, no markdown "
                                       "or bold.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "The concrete evidence supporting the "
                                       "finding (experiment, numbers, ablation, "
                                       "theorem). Do NOT restate the finding "
                                       "sentence. Plain prose, no markdown.",
                    },
                },
                "required": ["finding", "evidence"],
            },
        },
    },
    "required": ["title", "setting_and_object",
                 "concrete_detailed_setting", "key_findings"],
}

DEVELOPER_PROMPT = """\
You are an expert research-paper analyst. You will be given the raw LaTeX source \
of one paper. Produce a faithful summary in the Standard Paper Summary Format. \
Be precise, concrete, and grounded ONLY in the paper's content; never invent \
results, numbers, or baselines.

BE CONCISE AND HIGH-SIGNAL. Really work to keep only the main crux. Prefer fewer, \
denser items over exhaustive enumeration; merge or drop minor points; never pad a \
list to hit a count. Do not enumerate every dataset/benchmark/hyperparameter when \
a representative summary conveys the crux. A reader should grasp the essence \
quickly, not wade through a long list.

Guiding distinction: section 1 is the IRREDUCIBLE ESSENCE - the central object the \
paper contributes and the core problem it solves. Section 2 is the concrete context \
the authors chose LATER to instantiate and evaluate that object. Section 3 is what \
they found. Anything about where/how the object is evaluated, applied, or how well \
it does is NOT section 1. Follow these section rules exactly:

1. Main object & core problem - KEEP THIS EXTREMELY CRISP (the irreducible essence, \
nothing more).
   - MAIN OBJECT: the single central thing the paper contributes or studies, stated \
precisely enough that an expert recognizes the contribution. For an algorithm/\
method paper this is the ALGORITHM ITSELF - its defining mechanism, stated at the \
level of the ESSENTIAL IDEA: the distinctive computation that makes it this method \
(what it essentially does), not where it is applied. State it precisely enough to \
distinguish the method from its neighbors, but no finer - the essential idea, not \
the full formula. Implementation specifics that are not part of the core idea may \
be omitted (and belong in section 2 if they materially matter). For theory, the central \
result; for a benchmark/dataset, the artifact; for a software/system paper, the \
ARTIFACT and its KEY DESIGN PRINCIPLE(S).
   - CORE PROBLEM: in one sentence, the essential problem the object addresses (why \
it exists).
   - Bar: 1-3 crisp items total; from this alone a reader grasps 'what IS this \
paper, at its core?'.
   - Keep mere DEMONSTRATION context out of this crisp object. Judge by ROLE, not by \
item type: a thing is secondary when it is only where/how THIS paper's contribution \
is shown, and central when it IS the contribution. So for a method/algorithm/theory \
paper, the tasks, datasets, model families, scales, training/eval regimes, and \
metrics it is merely tested on - and the results it gets ('tested on X', 'applied \
to Y', 'outperforms Z') - are secondary and usually belong in section 2 / the key \
findings. But the SAME items can be the main object for a different paper: a dataset \
is the object of a dataset paper, a metric/protocol of an evaluation paper, the \
benchmark tasks of a benchmark paper. Include such a detail here only when it IS the \
paper's contribution (or is intrinsic to defining the object), not when it is just \
where the contribution is demonstrated.
   - Pick the single best category (Algorithm, Theory, Benchmark, Architecture, \
Dataset, Mechanism, Scaling law, Phenomenon, Evaluation protocol, ...).

2. Concrete detailed setting - the concrete decisions made later to instantiate \
and evaluate the idea, organized as GROUPS BY ASPECT (each group = one aspect \
heading plus a list of concrete detail items). Cover the relevant aspects: the \
exact datasets used (e.g. for a pretraining-optimizer paper, the specific \
pretraining corpus/mixture and tokenizer), model sizes/architectures, training & \
evaluation regimes and budgets, key hyperparameters, the baselines compared \
against, ablations run, and (for theory) the formal model, assumptions, and \
theorem/proof setup. This is where the secondary specifics belong, but stay \
selective: use 3-5 groups with only the decisions that materially matter, and do \
NOT exhaustively enumerate (e.g. give a representative set of datasets/benchmarks, \
not every one). A few crisp detail items per group.

3. Novel key findings - list from MOST to LEAST important; each genuinely novel \
to this paper. For each, write `finding` as ONE concise plain-prose sentence \
highlighting the finding, and `evidence` as the concrete support for it \
(experiment, numbers, ablation, theorem). Do NOT use bold/markdown, and do NOT \
restate the finding sentence inside `evidence`. Provide 3-5 findings, prioritized; \
only genuinely important ones.
"""


def build_user_prompt(arxiv_id: str, source_text: str) -> str:
    """Build the canonical per-paper user prompt used by every provider."""
    return (
        f"arXiv id: {arxiv_id}\n\n"
        "=== BEGIN PAPER LATEX SOURCE ===\n"
        f"{source_text}\n"
        "=== END PAPER LATEX SOURCE ==="
    )


def summarize(arxiv_id: str, source_text: str) -> dict:
    user = build_user_prompt(arxiv_id, source_text)
    return provider_client.structured(
        model=SUMMARY_MODEL,
        developer=DEVELOPER_PROMPT,
        user=user,
        schema=SUMMARY_SCHEMA,
        schema_name="paper_summary",
        max_output_tokens=16000,
        reasoning_effort="high",
    )


# --------------------------------------------------------------------------- #
# 4. Render Markdown
# --------------------------------------------------------------------------- #


def render_markdown(arxiv_id: str, s: dict) -> str:
    so = s["setting_and_object"]
    lines = [
        f"# {s.get('title') or arxiv_id}",
        "",
        f"*arXiv:{arxiv_id}*",
        "",
        "## 1. Setting and main object",
        "",
        f"*Category: {so['category']}*",
        "",
    ]
    for gi, group in enumerate(so["groups"], 1):
        lines.append(f"{gi}. **{group['aspect']}**")
        for di, detail in enumerate(group["details"], 1):
            lines.append(f"    {di}. {detail}")
    lines += [
        "",
        "## 2. Concrete detailed setting",
        "",
    ]
    for gi, group in enumerate(s["concrete_detailed_setting"], 1):
        lines.append(f"{gi}. **{group['aspect']}**")
        for di, detail in enumerate(group["details"], 1):
            lines.append(f"    {di}. {detail}")
    lines += [
        "",
        "## 3. Novel key findings",
        "",
    ]
    for i, f in enumerate(s["key_findings"], 1):
        finding = f["finding"].strip()
        lines.append(f"{i}. {finding}")
        evidence = f.get("evidence", "").strip()
        if evidence:
            lines.append(f"    {evidence}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_paper(arxiv_id: str, out_dir: Path, log=print) -> dict:
    log(f"=== {arxiv_id} ===")
    archive = download_source(arxiv_id, log)
    source_text = extract_source_text(archive, arxiv_id, log)
    log(f"  [llm] summarizing {len(source_text):,} chars with {SUMMARY_MODEL}")
    t0 = time.time()
    summary = summarize(arxiv_id, source_text)
    log(f"  [llm] done in {time.time() - t0:.1f}s; {len(summary['key_findings'])} findings")

    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"arxiv_id": arxiv_id, "model": SUMMARY_MODEL,
              "source_chars": len(source_text), "summary": summary}
    (out_dir / f"{arxiv_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2))
    md = render_markdown(arxiv_id, summary)
    (out_dir / f"{arxiv_id}.md").write_text(md)
    log(f"  [out] {out_dir / f'{arxiv_id}.md'}")
    return record


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arxiv_ids", nargs="+", help="arXiv ids, e.g. 2509.01229")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), type=Path)
    ap.add_argument("--trace", default=None,
                    help="optional JSONL trace path for the LLM client")
    args = ap.parse_args(argv)

    if args.trace:
        provider_client.set_trace(args.trace)

    failures: list[str] = []
    for arxiv_id in args.arxiv_ids:
        try:
            run_paper(arxiv_id, args.out_dir)
        except Exception as exc:  # noqa: BLE001 - keep going across papers
            failures.append(arxiv_id)
            print(f"  [error] {arxiv_id}: {type(exc).__name__}: {exc}")

    totals = provider_client.usage_totals()
    print(f"\nDONE. ok={len(args.arxiv_ids) - len(failures)}/{len(args.arxiv_ids)} "
          f"failed={failures or '-'} | calls={totals['calls']} "
          f"cost=${totals['cost_usd']:.2f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
