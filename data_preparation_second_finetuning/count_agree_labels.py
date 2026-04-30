#!/usr/bin/env python3
import argparse
import json
from collections import Counter


DEFAULT_INPUT = "data_preparation_second_finetuning/finetuning-2-ads-nonads.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Count ad/non-ad labels in a JSONL file.")
    parser.add_argument("input_file", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--label-key", default="ad_classification")
    args = parser.parse_args()

    counts = Counter()
    with open(args.input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            counts[row.get(args.label_key)] += 1

    print(f"ad: {counts['ad']}")
    print(f"non-ad: {counts['non-ad']}")


if __name__ == "__main__":
    main()
