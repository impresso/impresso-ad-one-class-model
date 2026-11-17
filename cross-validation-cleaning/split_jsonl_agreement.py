#!/usr/bin/env python3
"""
Split a JSONL file into two files based on whether `type` equals `predicted_type`.

Usage:
  python split_jsonl_agreement.py predictions_non_ads.jsonl

Outputs (same base name as input):
  <base>_agree.jsonl      — lines where type == predicted_type
  <base>_dissagree.jsonl  — lines where type != predicted_type  (spelling per request)

Notes:
- Gracefully handles missing or null fields. If either field is missing, the record goes to _dissagree.
- Accepts either `predicted_type` or `predictedType` as the prediction field.
- Prints a short summary to stderr when done.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def split_jsonl(input_path: Path) -> None:
    if not input_path.exists() or not input_path.is_file():
        print(f"[error] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    base = input_path.with_suffix("")
    out_agree = base.with_name(base.name + "_agree").with_suffix(".jsonl")
    # Keep the user's requested spelling: _dissagree
    out_disagree = base.with_name(base.name + "_dissagree").with_suffix(".jsonl")

    total = agree = disagree = skipped = 0

    # Open outputs once for efficiency
    with input_path.open("r", encoding="utf-8") as fin, \
         out_agree.open("w", encoding="utf-8") as fa, \
         out_disagree.open("w", encoding="utf-8") as fd:

        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] Skipping line {lineno}: JSON decode error: {e}", file=sys.stderr)
                skipped += 1
                continue

            t = obj.get("type")
            # Support both snake_case and camelCase for the prediction field
            pt: Optional[str] = obj.get("predicted_type", obj.get("predictedType"))

            # Missing fields are considered disagreement
            is_agree = (t is not None) and (pt is not None) and (t == pt)

            target = fa if is_agree else fd
            json.dump(obj, target, ensure_ascii=False)
            target.write("\n")

            if is_agree:
                agree += 1
            else:
                disagree += 1

    print(
        f"[done] total={total} agree={agree} disagree={disagree} skipped_invalid_json={skipped}\n"
        f"       wrote: {out_agree} and {out_disagree}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSONL by agreement of type and predicted_type")
    parser.add_argument("input", type=Path, help="Path to input .jsonl file")
    args = parser.parse_args()

    split_jsonl(args.input)


if __name__ == "__main__":
    main()
