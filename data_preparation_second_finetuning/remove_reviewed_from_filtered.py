#!/usr/bin/env python3
"""Remove already reviewed rows from the filtered JSONL by matching `id`.

Example:
python data_preparation_second_finetuning/remove_reviewed_from_filtered.py \
  data_preparation_second_finetuning/ads-nonads_inspection_sample_ft_disagree_filtered.jsonl \
  data_preparation_second_finetuning/ads-nonads_inspection_sample_ft_disagree_filtered_sampled_300.jsonl
"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove rows from a JSONL file when their ids are present in a reviewed/sample JSONL."
    )
    parser.add_argument("input_file", help="Full filtered JSONL file.")
    parser.add_argument("reviewed_file", help="JSONL file containing already reviewed rows.")
    parser.add_argument(
        "--output-file",
        help="Optional output path. Defaults to <input>_remaining.jsonl.",
    )
    parser.add_argument("--id-key", default="id", help="Row key for the article id.")
    return parser.parse_args()


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def derive_output_file(input_file):
    base, ext = os.path.splitext(os.path.abspath(input_file))
    if not ext:
        ext = ".jsonl"
    return f"{base}_remaining{ext}"


def main():
    args = parse_args()
    output_file = os.path.abspath(args.output_file) if args.output_file else derive_output_file(args.input_file)

    reviewed_ids = set()
    for row in iter_jsonl(args.reviewed_file):
        row_id = row.get(args.id_key)
        if isinstance(row_id, str) and row_id:
            reviewed_ids.add(row_id)

    kept = 0
    removed = 0
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        for row in iter_jsonl(args.input_file):
            row_id = row.get(args.id_key)
            if row_id in reviewed_ids:
                removed += 1
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Removed {removed} rows, kept {kept}. Output: {output_file}")


if __name__ == "__main__":
    main()
