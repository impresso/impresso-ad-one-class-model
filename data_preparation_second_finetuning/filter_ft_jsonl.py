#!/usr/bin/env python3
"""Filter JSONL rows by the contents of a text field.

By default, this writes a new sibling file with `_filtered` added.
Rows are dropped when:
- the text field contains no letters
- the text field is shorter than the configured minimum length
"""

import argparse
import json
import os
import tempfile
from typing import Any, Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a JSONL file by text quality."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional output path. Defaults to <input>_filtered.jsonl.",
    )
    parser.add_argument(
        "--text-key",
        default="ft",
        help="Row key containing the text to check (default: ft).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=100,
        help="Minimum character count for kept rows (default: 100).",
    )
    return parser.parse_args()


def has_letter(text: str) -> bool:
    return any(char.isalpha() for char in text)


def should_keep(row: Dict[str, Any], text_key: str, min_chars: int) -> Tuple[bool, str]:
    text = row.get(text_key)
    if not isinstance(text, str):
        return False, "missing_text"

    stripped = text.strip()
    if not has_letter(stripped):
        return False, "no_letters"
    if len(stripped) < min_chars:
        return False, "too_short"
    return True, "kept"


def derive_output_file(input_file: str) -> str:
    abs_input = os.path.abspath(input_file)
    base, ext = os.path.splitext(abs_input)
    if not ext:
        ext = ".jsonl"
    return f"{base}_filtered{ext}"


def filter_file(input_file: str, output_file: str, text_key: str, min_chars: int) -> Dict[str, int]:
    stats = {
        "seen": 0,
        "kept": 0,
        "missing_text": 0,
        "no_letters": 0,
        "skipped_comments": 0,
        "too_short": 0,
    }

    with open(input_file, "r", encoding="utf-8") as src, open(output_file, "w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            if stripped_line.startswith("#"):
                stats["skipped_comments"] += 1
                continue

            try:
                row = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_file}:{line_no}: {exc}") from exc

            stats["seen"] += 1
            keep, reason = should_keep(row, text_key=text_key, min_chars=min_chars)
            if keep:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["kept"] += 1
            else:
                stats[reason] += 1

    return stats


def main() -> None:
    args = parse_args()
    if args.min_chars <= 0:
        raise SystemExit("--min-chars must be greater than 0")

    input_path = os.path.abspath(args.input_file)
    output_file = os.path.abspath(args.output_file) if args.output_file else derive_output_file(input_path)
    parent_dir = os.path.dirname(output_file) or "."
    os.makedirs(parent_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".filter_ft_", suffix=".jsonl", dir=parent_dir)
    os.close(fd)
    try:
        stats = filter_file(input_path, temp_path, args.text_key, args.min_chars)
        os.replace(temp_path, output_file)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print(
        f"Kept {stats['kept']} of {stats['seen']} rows. "
        f"Dropped missing_text={stats['missing_text']} "
        f"no_letters={stats['no_letters']} too_short={stats['too_short']} "
        f"skipped_comments={stats['skipped_comments']}. "
        f"Output: {output_file}"
    )


if __name__ == "__main__":
    main()
