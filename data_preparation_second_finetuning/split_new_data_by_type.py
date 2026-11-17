#!/usr/bin/env python3
"""Split new_data_all_deduplicated.jsonl into ad and non-ad JSONL files."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = SCRIPT_DIR / "new_data_all_deduplicated.jsonl"
DEFAULT_ADS_OUTPUT_JSONL = SCRIPT_DIR / "new_ads.jsonl"
DEFAULT_NON_ADS_OUTPUT_JSONL = SCRIPT_DIR / "new_non_ads.jsonl"

Row = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a JSONL file into ad and non-ad JSONL files based on row['type']."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_INPUT_JSONL,
        help=f"Input JSONL to split (default: {DEFAULT_INPUT_JSONL})",
    )
    parser.add_argument(
        "--ads-output-jsonl",
        type=Path,
        default=DEFAULT_ADS_OUTPUT_JSONL,
        help=f"Output JSONL for ad rows (default: {DEFAULT_ADS_OUTPUT_JSONL})",
    )
    parser.add_argument(
        "--non-ads-output-jsonl",
        type=Path,
        default=DEFAULT_NON_ADS_OUTPUT_JSONL,
        help=f"Output JSONL for non-ad rows (default: {DEFAULT_NON_ADS_OUTPUT_JSONL})",
    )
    return parser.parse_args()


def parse_json_line(line: str, path: Path, line_no: int) -> Row:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc

    if not isinstance(row, dict):
        raise ValueError(f"Expected a JSON object at {path}:{line_no}")

    return row


def normalize_type(value: Any, path: Path, line_no: int) -> str:
    label = str(value or "").strip().lower().replace("_", "-")
    if label == "ad":
        return "ad"
    if label in {"non-ad", "nonad"}:
        return "non-ad"
    raise ValueError(f"Invalid type value at {path}:{line_no}: {value!r}")


def split_jsonl(input_path: Path, ads_output_path: Path, non_ads_output_path: Path) -> Dict[str, int]:
    ads_output_path.parent.mkdir(parents=True, exist_ok=True)
    non_ads_output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"input_rows": 0, "ad_rows": 0, "non_ad_rows": 0}

    with input_path.open("r", encoding="utf-8") as input_handle, ads_output_path.open(
        "w", encoding="utf-8"
    ) as ads_handle, non_ads_output_path.open("w", encoding="utf-8") as non_ads_handle:
        for line_no, line in enumerate(input_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            row = parse_json_line(stripped, input_path, line_no)
            row_type = normalize_type(row.get("type"), input_path, line_no)
            output_line = json.dumps(row, ensure_ascii=False) + "\n"

            counts["input_rows"] += 1
            if row_type == "ad":
                ads_handle.write(output_line)
                counts["ad_rows"] += 1
            else:
                non_ads_handle.write(output_line)
                counts["non_ad_rows"] += 1

    return counts


def main() -> None:
    args = parse_args()
    counts = split_jsonl(
        args.input_jsonl,
        args.ads_output_jsonl,
        args.non_ads_output_jsonl,
    )

    print(f"Input rows: {counts['input_rows']}")
    print(f"Ad rows: {counts['ad_rows']}")
    print(f"Non-ad rows: {counts['non_ad_rows']}")
    print(f"Ads output file: {args.ads_output_jsonl}")
    print(f"Non-ads output file: {args.non_ads_output_jsonl}")


if __name__ == "__main__":
    main()
