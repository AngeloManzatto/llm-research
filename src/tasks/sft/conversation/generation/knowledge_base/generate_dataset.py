# -*- coding: utf-8 -*-
"""
generate_dataset.py — runs the full relation-table generator.

Usage:
    python generate_dataset.py --out kb_generated.jsonl

Every relation's fact table is fully cross-joined against its
templates, for every language it defines. Output size is fully
predictable before generation — printed per relation, so you can see
exactly how many rows each one contributes and where the real
diversity headroom is (more facts and/or more templates), rather than
discovering the count as a surprise afterward.

This is deterministic generation, not sampling — running this twice
produces byte-identical output. Deduplication tooling (dedup_scan.py /
dedup_prune.py) is still worth running afterward if this output gets
merged with anything else (e.g. the existing LLM-generated corpus),
but this generator cannot produce internal duplicates on its own by
construction — every (subject, template) pair is used exactly once.
"""

from __future__ import annotations

import argparse
import json

from relations import RELATIONS, generate_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--languages", nargs="+", default=["en", "pt"])
    args = parser.parse_args()

    all_rows = []
    print(f"{'Relation':<16} {'Language':<10} {'Facts':>7} {'Templates':>11} {'Rows':>7}")
    print("-" * 55)

    for relation in RELATIONS:
        for lang in args.languages:
            if lang not in relation.facts:
                continue
            rows = generate_rows(relation, lang)
            all_rows.extend(rows)
            print(f"{relation.name:<16} {lang:<10} {relation.n_facts(lang):>7} "
                  f"{relation.n_templates(lang):>11} {len(rows):>7}")

    print("-" * 55)
    print(f"{'TOTAL':<16} {'':<10} {'':<7} {'':<11} {len(all_rows):>7}")

    with open(args.out, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
