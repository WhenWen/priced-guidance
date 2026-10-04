"""Validate target-pack integrity and every gold record without model calls."""
from __future__ import annotations
import argparse
import json
from tech_tree_arena.targets import load_target_pack

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packs", nargs="+")
    args = parser.parse_args()
    for path in args.packs:
        pack = load_target_pack(path)
        ids = pack.target_ids()
        if not ids:
            raise ValueError(f"empty target pack: {path}")
        for target_id in ids:
            pack.load(target_id)
        print(json.dumps({"pack": pack.name, "targets": len(ids), "status": "ok"}))

if __name__ == "__main__":
    main()
