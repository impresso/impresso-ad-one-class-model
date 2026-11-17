#!/usr/bin/env python3
import argparse
import json


DEFAULT_INPUT = "fine-tuning-data/finetuning-2-ads-nonads.jsonl"
DEFAULT_OUTPUT = "fine-tuning-data/finetuning-2-ads-nonads_converted.jsonl"


def normalize_label(value):
    label = str(value or "").strip().lower().replace("_", "-")
    if label == "ad":
        return "ad"
    if label in {"non-ad", "nonad"}:
        return "non-ad"
    raise ValueError(f"Invalid ad_classification value: {value!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert finetuning-2 ad/non-ad JSONL to fine-tuning-data format."
    )
    parser.add_argument("input_jsonl", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("output_jsonl", nargs="?", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with open(args.input_jsonl, "r", encoding="utf-8") as input_handle, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as output_handle:
        for line_no, line in enumerate(input_handle, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            article_id = str(row.get("id") or "").strip()
            text = str(row.get("ft") or "").strip()
            if not article_id or not text:
                continue

            output_row = {
                "id": article_id,
                "lg": None,
                "ft": text,
                "type": normalize_label(row.get("ad_classification")),
            }
            output_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
