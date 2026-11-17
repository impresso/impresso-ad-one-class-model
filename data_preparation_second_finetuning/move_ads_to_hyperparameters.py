#!/usr/bin/env python3
"""Randomly move rows from ads fine-tuning data into ads hyperparameter data."""

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_JSONL = SCRIPT_DIR / "non_ads_8823_finetuning.jsonl"
DEFAULT_TARGET_JSONL = SCRIPT_DIR / "non_ads_100_for_hyperparameters.jsonl"
DEFAULT_MOVE_COUNT = 200

Row = Dict[str, Any]
LineRow = Tuple[str, Row]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly move rows from ads_8823_finetuning.jsonl to "
            "ads_100_for_hyperparameters.jsonl. Moved rows are appended to the "
            "target and removed from the source."
        )
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=DEFAULT_SOURCE_JSONL,
        help=f"Source JSONL to sample from (default: {DEFAULT_SOURCE_JSONL})",
    )
    parser.add_argument(
        "--target-jsonl",
        type=Path,
        default=DEFAULT_TARGET_JSONL,
        help=f"Target JSONL to append to (default: {DEFAULT_TARGET_JSONL})",
    )
    parser.add_argument(
        "--move-count",
        type=int,
        default=DEFAULT_MOVE_COUNT,
        help=f"Number of rows to move (default: {DEFAULT_MOVE_COUNT})",
    )
    parser.add_argument(
        "--id-key",
        default="id",
        help="JSON key used to avoid moving rows already in the target (default: id)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing any files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create timestamped backups before modifying files.",
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


def parse_row_id(row: Row, id_key: str, path: Path, line_no: int) -> str:
    row_id = row.get(id_key)
    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError(f"Missing or invalid {id_key!r} at {path}:{line_no}")
    return row_id.strip()


def load_jsonl(path: Path) -> List[LineRow]:
    rows: List[LineRow] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            rows.append((stripped, parse_json_line(stripped, path, line_no)))

    return rows


def row_ids(rows: Sequence[LineRow], id_key: str, path: Path) -> Set[str]:
    ids: Set[str] = set()
    for index, (_, row) in enumerate(rows, start=1):
        row_id = parse_row_id(row, id_key, path, index)
        if row_id in ids:
            raise ValueError(f"Duplicate {id_key!r} found in {path}: {row_id}")
        ids.add(row_id)
    return ids


def sample_move_indices(
    source_rows: Sequence[LineRow],
    target_ids: Set[str],
    move_count: int,
    id_key: str,
    source_path: Path,
    seed: int | None,
) -> Set[int]:
    if move_count <= 0:
        raise ValueError("--move-count must be greater than 0")

    eligible_indices: List[int] = []
    for index, (_, row) in enumerate(source_rows):
        row_id = parse_row_id(row, id_key, source_path, index + 1)
        if row_id not in target_ids:
            eligible_indices.append(index)

    if len(eligible_indices) < move_count:
        raise ValueError(
            f"Requested {move_count} rows, but only {len(eligible_indices)} eligible rows "
            "are available after excluding IDs already in the target."
        )

    rng = random.Random(seed)
    return set(rng.sample(eligible_indices, move_count))


def backup_file(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak-{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def write_moved_files(
    source_path: Path,
    target_path: Path,
    source_rows: Sequence[LineRow],
    move_indices: Set[int],
) -> None:
    kept_source_lines = [
        line for index, (line, _) in enumerate(source_rows) if index not in move_indices
    ]
    moved_lines = [
        line for index, (line, _) in enumerate(source_rows) if index in move_indices
    ]

    with source_path.open("w", encoding="utf-8") as source_handle:
        for line in kept_source_lines:
            source_handle.write(line + "\n")

    with target_path.open("a", encoding="utf-8") as target_handle:
        for line in moved_lines:
            target_handle.write(line + "\n")


def main() -> None:
    args = parse_args()

    source_rows = load_jsonl(args.source_jsonl)
    target_rows = load_jsonl(args.target_jsonl)
    target_ids = row_ids(target_rows, args.id_key, args.target_jsonl)
    move_indices = sample_move_indices(
        source_rows,
        target_ids,
        args.move_count,
        args.id_key,
        args.source_jsonl,
        args.seed,
    )

    print(f"Source rows before: {len(source_rows)}")
    print(f"Target rows before: {len(target_rows)}")
    print(f"Rows to move: {len(move_indices)}")
    print(f"Source rows after: {len(source_rows) - len(move_indices)}")
    print(f"Target rows after: {len(target_rows) + len(move_indices)}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")

    if args.dry_run:
        print("Dry run: no files were modified.")
        return

    if not args.no_backup:
        source_backup = backup_file(args.source_jsonl)
        target_backup = backup_file(args.target_jsonl)
        print(f"Source backup: {source_backup}")
        print(f"Target backup: {target_backup}")

    write_moved_files(args.source_jsonl, args.target_jsonl, source_rows, move_indices)
    print(f"Updated source file: {args.source_jsonl}")
    print(f"Updated target file: {args.target_jsonl}")


if __name__ == "__main__":
    main()
