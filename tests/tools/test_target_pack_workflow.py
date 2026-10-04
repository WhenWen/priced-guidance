import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.target_packs import build_development_pack as builder
from tools.target_packs import opus5_batch_summary as opus
from tools.target_packs import paper_summary
from tech_tree_arena.targets import load_target_pack

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "src/tech_tree_arena/data/target_packs/development40"

def prepare(tmp_path):
    source = next((PUBLIC / "secret/gold").glob("*.json"))
    record = json.loads(source.read_text())
    ident = record["arxiv_id"]
    # Exercise exactly the Opus response -> JSON summary -> target pack boundary.
    message = SimpleNamespace(id="test-message", stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 10},
        content=[SimpleNamespace(type="text", text=json.dumps(record["summary"]))])
    opus._write_outputs(out_dir=tmp_path / "summaries", paper={"arxiv_id": ident},
        source_metadata={"fixture": True}, batch_id="test-batch", message=message,
        model="claude-opus-5", effort="high", service_tier="standard", prices=opus.STANDARD_PRICES)
    selection = tmp_path / "papers.jsonl"
    selection.write_text(json.dumps({"arxiv_id": ident}) + "\n")
    args = ["--selection", str(selection), "--gold-dir", str(tmp_path / "summaries"),
        "--public-source-pack", str(PUBLIC), "--name", "new-cases", "--out-dir", str(tmp_path / "pack")]
    return ident, args

def test_opus_summary_becomes_loadable_target_pack(tmp_path):
    ident, args = prepare(tmp_path)
    assert builder.main(args) == 0
    pack = load_target_pack(tmp_path / "pack")
    assert pack.target_ids() == (ident,)
    assert pack.load(ident)["model"] == "claude-opus-5"
    assert set(pack.public_resources()) == {"taxonomy", "taxonomy_rankings"}
    # Never overwrite an existing benchmark on a retry.
    before = (tmp_path / "pack/pack.toml").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        builder.main(args)
    assert (tmp_path / "pack/pack.toml").read_bytes() == before
    # Tampering with the evaluator target must be detected.
    gold = tmp_path / "pack/secret/gold" / f"{ident}.json"
    gold.write_text(gold.read_text() + " ")
    with pytest.raises(Exception, match="gold-tree hash"):
        load_target_pack(tmp_path / "pack")

@pytest.mark.parametrize("ident", ["../../escape", "2604.27351/other", "", "name"])
def test_invalid_target_ids_rejected_before_writing(tmp_path, ident):
    _, args = prepare(tmp_path)
    (tmp_path / "papers.jsonl").write_text(json.dumps({"arxiv_id": ident}))
    with pytest.raises(ValueError, match="arXiv IDs"):
        builder.main(args)
    assert not (tmp_path / "pack").exists()

def test_duplicate_target_ids_rejected(tmp_path):
    ident, args = prepare(tmp_path)
    (tmp_path / "papers.jsonl").write_text((json.dumps({"arxiv_id": ident}) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        builder.main(args)
    assert not (tmp_path / "pack").exists()

def test_source_archive_cannot_escape_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_summary, "CACHE_DIR", tmp_path / "cache")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../../escape.tex")
        data = b"forbidden"
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match="escapes"):
        paper_summary.extract_source_text(archive, "2604.27351")
    assert not (tmp_path / "escape.tex").exists()

def test_tex_include_cannot_read_outside_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "private.tex").write_text("private")
    entry = source / "main.tex"
    entry.write_text(r"\input{../private}")
    with pytest.raises(ValueError, match="escapes"):
        paper_summary._flatten_tex(entry, set(), [])

def test_standard_and_batch_commands_parse():
    for tier in ["standard", "batch"]:
        args = opus.build_parser().parse_args(["--papers", "papers.jsonl", "--out-dir", "summaries",
            "--model", "claude-opus-5", "--effort", "high", "--service-tier", tier])
        assert args.model == "claude-opus-5" and args.service_tier == tier
