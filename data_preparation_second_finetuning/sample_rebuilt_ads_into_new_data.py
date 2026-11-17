#!/usr/bin/env python3
"""Sample new rebuilt ad rows and append them to the current fine-tuning data."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_JSONL = SCRIPT_DIR / "rebuilt_one_ad_per_newspaper_year.jsonl"
DEFAULT_CURRENT_JSONL = SCRIPT_DIR / "finetuning-2-ads-nonads.jsonl"
DEFAULT_OUTPUT_JSONL = SCRIPT_DIR / "new_data_all.jsonl"
DEFAULT_SAMPLE_SIZE = 2800

Row = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample rows from rebuilt_one_ad_per_newspaper_year.jsonl whose IDs are "
            "not already in finetuning-2-ads-nonads.jsonl, then append them to the "
            "current fine-tuning data."
        )
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=DEFAULT_SOURCE_JSONL,
        help=f"Input pool to sample from (default: {DEFAULT_SOURCE_JSONL})",
    )
    parser.add_argument(
        "--current-jsonl",
        type=Path,
        default=DEFAULT_CURRENT_JSONL,
        help=f"Existing fine-tuning JSONL to append to (default: {DEFAULT_CURRENT_JSONL})",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help=f"Combined output JSONL (default: {DEFAULT_OUTPUT_JSONL})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of new rows to sample (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--id-key",
        default="id",
        help="JSON key used to identify duplicate rows (default: id)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible sampling.",
    )
    return parser.parse_args()


def load_current_lines_and_ids(path: Path, id_key: str) -> Tuple[List[str], Set[str]]:
    lines: List[str] = []
    ids: Set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            row = parse_json_line(stripped, path, line_no)
            row_id = parse_row_id(row, id_key, path, line_no)
            ids.add(row_id)
            lines.append(stripped)

    return lines, ids


def load_new_candidate_rows(path: Path, existing_ids: Set[str], id_key: str) -> List[Row]:
    candidates: List[Row] = []
    seen_source_ids: Set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            row = parse_json_line(stripped, path, line_no)
            row_id = parse_row_id(row, id_key, path, line_no)

            if row_id in existing_ids or row_id in seen_source_ids:
                continue

            seen_source_ids.add(row_id)
            candidates.append(row)

    return candidates


def parse_json_line(line: str, path: Path, line_no: int) -> Row:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc

    if not isinstance(row, dict):
        raise ValueError(f"Expected a JSON object at {path}:{line_no}")

    return row


def parse_row_id(row: Row, id_key: str, path: Path, line_no: int) -> str:
    row_id = row.get(id_key)
    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError(f"Missing or invalid {id_key!r} at {path}:{line_no}")
    return row_id.strip()


def sample_rows(rows: List[Row], sample_size: int, seed: int | None) -> List[Row]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be greater than 0")
    if len(rows) < sample_size:
        raise ValueError(
            f"Requested {sample_size} rows, but only {len(rows)} new rows are available."
        )

    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def write_combined_jsonl(current_lines: Iterable[str], sampled_rows: Iterable[Row], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in current_lines:
            handle.write(line.rstrip("\n") + "\n")
        for row in sampled_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    current_lines, existing_ids = load_current_lines_and_ids(args.current_jsonl, args.id_key)
    candidates = load_new_candidate_rows(args.source_jsonl, existing_ids, args.id_key)
    sampled_rows = sample_rows(candidates, args.sample_size, args.seed)
    write_combined_jsonl(current_lines, sampled_rows, args.output_jsonl)

    print(f"Existing rows: {len(current_lines)}")
    print(f"New candidate rows: {len(candidates)}")
    print(f"Sampled rows: {len(sampled_rows)}")
    print(f"Output rows: {len(current_lines) + len(sampled_rows)}")
    print(f"Output file: {args.output_jsonl}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
