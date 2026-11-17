#!/usr/bin/env python3
"""Remove rows from new_data_all.jsonl that already appear in held-out/reference files."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = SCRIPT_DIR / "new_data_all.jsonl"
DEFAULT_OUTPUT_JSONL = SCRIPT_DIR / "new_data_all_deduplicated.jsonl"
DEFAULT_REFERENCE_JSONLS = [
    SCRIPT_DIR / "ads_100_for_hyperparameters.jsonl",
    SCRIPT_DIR / "non_ads_100_for_hyperparameters.jsonl",
    SCRIPT_DIR / "ads_100_for_testing.jsonl",
    SCRIPT_DIR / "non_ads_100_for_testing.jsonl",
    SCRIPT_DIR / "ads_3800_finetuning.jsonl",
    SCRIPT_DIR / "non_ads_3800_finetuning.jsonl",
]

Row = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write rows from new_data_all.jsonl whose IDs do not appear in any "
            "reference JSONL file."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_INPUT_JSONL,
        help=f"Input JSONL to filter (default: {DEFAULT_INPUT_JSONL})",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help=f"Filtered output JSONL (default: {DEFAULT_OUTPUT_JSONL})",
    )
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        action="append",
        dest="reference_jsonls",
        help=(
            "Reference JSONL to exclude from. Repeat this option to override the "
            "default reference file list."
        ),
    )
    parser.add_argument(
        "--id-key",
        default="id",
        help="JSON key used to identify duplicate rows (default: id)",
    )
    parser.add_argument(
        "--allow-missing-reference-files",
        action="store_true",
        help="Skip missing reference files instead of failing.",
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


def validate_reference_files(paths: Iterable[Path], allow_missing: bool) -> List[Path]:
    existing_paths: List[Path] = []
    missing_paths: List[Path] = []

    for path in paths:
        if path.exists():
            existing_paths.append(path)
        else:
            missing_paths.append(path)

    if missing_paths and not allow_missing:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing reference JSONL file(s):\n"
            f"{missing}\n"
            "Create these files, pass the correct paths with --reference-jsonl, "
            "or rerun with --allow-missing-reference-files."
        )

    return existing_paths


def load_reference_ids(paths: Iterable[Path], id_key: str) -> Set[str]:
    ids: Set[str] = set()

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                row = parse_json_line(stripped, path, line_no)
                ids.add(parse_row_id(row, id_key, path, line_no))

    return ids


def write_filtered_rows(input_path: Path, output_path: Path, reference_ids: Set[str], id_key: str) -> Dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        "input_rows": 0,
        "written_rows": 0,
        "skipped_reference_rows": 0,
        "skipped_input_duplicate_rows": 0,
    }
    written_ids: Set[str] = set()

    with input_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_no, line in enumerate(input_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            row = parse_json_line(stripped, input_path, line_no)
            row_id = parse_row_id(row, id_key, input_path, line_no)
            counts["input_rows"] += 1

            if row_id in reference_ids:
                counts["skipped_reference_rows"] += 1
                continue

            if row_id in written_ids:
                counts["skipped_input_duplicate_rows"] += 1
                continue

            written_ids.add(row_id)
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["written_rows"] += 1

    return counts


def main() -> None:
    args = parse_args()
    reference_paths = args.reference_jsonls or DEFAULT_REFERENCE_JSONLS
    existing_reference_paths = validate_reference_files(
        reference_paths, args.allow_missing_reference_files
    )
    reference_ids = load_reference_ids(existing_reference_paths, args.id_key)
    counts = write_filtered_rows(args.input_jsonl, args.output_jsonl, reference_ids, args.id_key)

    print(f"Input rows: {counts['input_rows']}")
    print(f"Reference files used: {len(existing_reference_paths)}")
    print(f"Reference IDs: {len(reference_ids)}")
    print(f"Skipped rows found in references: {counts['skipped_reference_rows']}")
    print(f"Skipped duplicate rows within input: {counts['skipped_input_duplicate_rows']}")
    print(f"Written rows: {counts['written_rows']}")
    print(f"Output file: {args.output_jsonl}")


if __name__ == "__main__":
    main()
