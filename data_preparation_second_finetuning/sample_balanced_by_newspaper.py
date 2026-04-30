#!/usr/bin/env python3
"""Sample JSONL rows as evenly across newspapers as possible.

The newspaper code is read from the first segment of the `id`, e.g.
`AGMO-1817-08-31-a-i0010` -> newspaper=`AGMO`, year=`1817`.

Sampling is balanced in two steps:
1. Allocate the requested sample size across newspapers as evenly as possible.
2. Within each newspaper, spread picks across years in a round-robin way.
"""

"""
python data_preparation_second_finetuning/sample_balanced_by_newspaper.py \
  data_preparation_second_finetuning/ads-nonads_inspection_sample_ft_disagree_filtered.jsonl \
  --sample-size 400


"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict, deque
from typing import Any, Deque, Dict, List, Tuple


Row = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample JSONL rows, balanced by newspaper as much as possible."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        required=True,
        help="Number of rows to sample.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional output path. Defaults to <input>_sampled_<N>.jsonl.",
    )
    parser.add_argument(
        "--id-key",
        default="id",
        help="Row key containing the content-item id (default: id).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    return parser.parse_args()


def derive_output_file(input_file: str, sample_size: int) -> str:
    abs_input = os.path.abspath(input_file)
    base, ext = os.path.splitext(abs_input)
    if not ext:
        ext = ".jsonl"
    return f"{base}_sampled_{sample_size}{ext}"


def parse_newspaper_year(content_id: str) -> Tuple[str, str]:
    parts = content_id.split("-")
    if len(parts) < 2:
        raise ValueError(f"Could not parse newspaper/year from id={content_id!r}")
    return parts[0], parts[1]


def load_rows(input_file: str, id_key: str) -> List[Row]:
    rows: List[Row] = []
    with open(input_file, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_file}:{line_no}: {exc}") from exc

            row_id = row.get(id_key)
            if not isinstance(row_id, str) or not row_id.strip():
                raise ValueError(f"Missing or invalid {id_key!r} at {input_file}:{line_no}")

            newspaper, year = parse_newspaper_year(row_id)
            row["_sample_newspaper"] = newspaper
            row["_sample_year"] = year
            rows.append(row)

    return rows


def allocate_balanced_counts(rows_by_newspaper: Dict[str, List[Row]], sample_size: int, rng: random.Random) -> Dict[str, int]:
    available = {paper: len(rows) for paper, rows in rows_by_newspaper.items()}
    allocation = {paper: 0 for paper in rows_by_newspaper}
    active = [paper for paper, count in available.items() if count > 0]

    if sample_size > sum(available.values()):
        sample_size = sum(available.values())

    rng.shuffle(active)
    while sample_size > 0 and active:
        next_active: List[str] = []
        for paper in active:
            if sample_size == 0:
                break
            if allocation[paper] < available[paper]:
                allocation[paper] += 1
                sample_size -= 1
            if allocation[paper] < available[paper]:
                next_active.append(paper)
        active = next_active

    return allocation


def sample_within_newspaper(rows: List[Row], quota: int, rng: random.Random) -> List[Row]:
    rows_by_year: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        rows_by_year[row["_sample_year"]].append(row)

    year_queues: Dict[str, Deque[Row]] = {}
    years = list(rows_by_year)
    rng.shuffle(years)
    for year in years:
        year_rows = rows_by_year[year]
        rng.shuffle(year_rows)
        year_queues[year] = deque(year_rows)

    selected: List[Row] = []
    active_years = years[:]
    while len(selected) < quota and active_years:
        next_active_years: List[str] = []
        for year in active_years:
            queue = year_queues[year]
            if queue and len(selected) < quota:
                selected.append(queue.popleft())
            if queue:
                next_active_years.append(year)
        active_years = next_active_years

    return selected


def sample_rows(rows: List[Row], sample_size: int, rng: random.Random) -> List[Row]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be greater than 0")
    if not rows:
        return []

    rows_by_newspaper: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        rows_by_newspaper[row["_sample_newspaper"]].append(row)

    allocation = allocate_balanced_counts(rows_by_newspaper, sample_size, rng)
    selected: List[Row] = []
    newspapers = list(rows_by_newspaper)
    rng.shuffle(newspapers)

    for newspaper in newspapers:
        quota = allocation[newspaper]
        if quota <= 0:
            continue
        selected.extend(sample_within_newspaper(rows_by_newspaper[newspaper], quota, rng))

    rng.shuffle(selected)
    return selected


def strip_helper_fields(rows: List[Row]) -> List[Row]:
    cleaned: List[Row] = []
    for row in rows:
        cleaned_row = dict(row)
        cleaned_row.pop("_sample_newspaper", None)
        cleaned_row.pop("_sample_year", None)
        cleaned.append(cleaned_row)
    return cleaned


def write_rows(rows: List[Row], output_file: str) -> None:
    parent_dir = os.path.dirname(output_file) or "."
    os.makedirs(parent_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    rows = load_rows(args.input_file, args.id_key)
    available = len(rows)
    requested = args.sample_size
    actual_sample_size = min(requested, available)

    selected = sample_rows(rows, actual_sample_size, rng)
    output_file = os.path.abspath(args.output_file) if args.output_file else derive_output_file(args.input_file, actual_sample_size)
    write_rows(strip_helper_fields(selected), output_file)

    sampled_by_newspaper = Counter(row["_sample_newspaper"] for row in selected)
    print(
        f"Sampled {len(selected)} rows from {available} available rows across "
        f"{len(sampled_by_newspaper)} newspapers. Output: {output_file}"
    )
    print(f"Requested={requested} Actual={len(selected)} Seed={args.seed}")
    print(f"Top newspaper counts: {sampled_by_newspaper.most_common(15)}")


if __name__ == "__main__":
    main()
