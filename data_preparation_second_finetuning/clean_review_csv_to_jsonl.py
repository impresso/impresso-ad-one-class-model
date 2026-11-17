#!/usr/bin/env python3
import argparse
import csv
import json


"""

python data_preparation_second_finetuning/clean_review_csv_to_jsonl.py \
  "data_preparation_second_finetuning/first_review.csv" \
  data_preparation_second_finetuning/first_human_annotations_review.jsonl

"""


AD_DECISION = "ann_ads_public_notices"
TRUE_VALUES = {"true", "1", "yes", "y"}


def is_true(value):
    return str(value or "").strip().lower() in TRUE_VALUES


def main():
    parser = argparse.ArgumentParser(
        description="Convert reviewed CSV rows to ad/non-ad JSONL training rows."
    )
    parser.add_argument("input_csv", help="Input CSV review file.")
    parser.add_argument("output_jsonl", help="Output JSONL file.")
    args = parser.parse_args()

    with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as csv_handle, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as jsonl_handle:
        reader = csv.DictReader(csv_handle)

        for row in reader:
            if is_true(row.get("Unsure")):
                continue

            article_id = str(row.get("id") or "").strip()
            text = str(row.get("Text") or "").strip()
            decision = str(row.get("Your Decision") or "").strip()

            if not article_id or not text:
                continue

            output_row = {
                "id": article_id,
                "ad_classification": "ad" if decision == AD_DECISION else "non-ad",
                "ft": text,
            }
            jsonl_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
