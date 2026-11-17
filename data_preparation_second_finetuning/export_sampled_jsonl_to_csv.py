#!/usr/bin/env python3
"""Export sampled JSONL rows to a CSV review sheet."""

import argparse
import csv
import json
import os
from typing import Dict, Iterable, Any

"""
python data_preparation_second_finetuning/export_sampled_jsonl_to_csv.py \
  data_preparation_second_finetuning/ads-nonads_inspection_sample_ft_disagree_filtered_sampled_300.jsonl


"""


ARTICLE_URL_TEMPLATE = "https://dev.impresso-project.ch/app/article/{article_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert sampled JSONL rows into a CSV with article links."
    )
    parser.add_argument(
        "input_file",
        help="Path to the sampled JSONL file.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional output CSV path. Defaults to <input>.csv with _review suffix.",
    )
    parser.add_argument(
        "--id-key",
        default="id",
        help="Input key containing the article id (default: id).",
    )
    parser.add_argument(
        "--text-key",
        default="ft",
        help="Input key containing the article text (default: ft).",
    )
    parser.add_argument(
        "--our-prediction-key",
        default="ad_classification",
        help="Input key containing your model prediction (default: ad_classification).",
    )
    parser.add_argument(
        "--openai-prediction-key",
        default="openai_classification",
        help="Input key containing the OpenAI prediction (default: openai_classification).",
    )
    return parser.parse_args()


def derive_output_file(input_file: str) -> str:
    abs_input = os.path.abspath(input_file)
    base, _ = os.path.splitext(abs_input)
    return f"{base}_review.csv"


def iter_rows(input_file: str) -> Iterable[Dict[str, Any]]:
    with open(input_file, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_file}:{line_no}: {exc}") from exc


def main() -> None:
    args = parse_args()
    output_file = os.path.abspath(args.output_file) if args.output_file else derive_output_file(args.input_file)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    fieldnames = ["ft", "id", "our_model_prediction", "openai_prediction", "article_link"]
    written = 0

    with open(output_file, "w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in iter_rows(args.input_file):
            article_id = row.get(args.id_key)
            if not isinstance(article_id, str) or not article_id.strip():
                raise ValueError(f"Missing or invalid {args.id_key!r} in input row: {row}")

            writer.writerow(
                {
                    "ft": row.get(args.text_key, ""),
                    "id": article_id,
                    "our_model_prediction": row.get(args.our_prediction_key, ""),
                    "openai_prediction": row.get(args.openai_prediction_key, ""),
                    "article_link": ARTICLE_URL_TEMPLATE.format(article_id=article_id),
                }
            )
            written += 1

    print(f"Wrote {written} rows to {output_file}")


if __name__ == "__main__":
    main()
