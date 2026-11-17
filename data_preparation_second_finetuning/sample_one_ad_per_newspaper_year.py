#!/usr/bin/env python3
"""Sample one ad per newspaper-year from rebuilt content-item files on S3.

The script streams JSONL files under an S3 root, keeps one pseudo-random ad row
per newspaper-year, and writes rows in the fine-tuning format:
{"id": "...", "lg": "...", "ft": "...", "type": "ad"}

Example:
python data_preparation_second_finetuning/sample_one_ad_per_newspaper_year.py \
  --s3-root s3://122-rebuilt-final/ \
  --output fine-tuning-data/rebuilt_one_ad_per_newspaper_year.jsonl \
  --seed 42
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from smart_open import open as smart_open


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
# cookbook helpers (common.py, s3_aggregator.py) live in the cookbook submodule
COOKBOOK_LIB_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "cookbook", "lib")
if COOKBOOK_LIB_DIR not in sys.path:
    sys.path.append(COOKBOOK_LIB_DIR)

import s3_aggregator as agg  # noqa: E402
from common import extract_newspaper_id, extract_year  # noqa: E402


LOG = logging.getLogger(__name__)

DEFAULT_S3_ROOT = "s3://122-rebuilt-final/"
SUPPORTED_SUFFIXES = (".jsonl", ".jsonl.bz2", ".jsonl.gz")


@dataclass
class Candidate:
    score: int
    row: Dict[str, Any]


@dataclass
class Counters:
    files_seen: int = 0
    files_scanned: int = 0
    lines_read: int = 0
    ad_rows_seen: int = 0
    sampled_groups: int = 0
    skipped_invalid_json: int = 0
    skipped_invalid_id: int = 0
    skipped_missing_text: int = 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample one random ad per newspaper-year from rebuilt S3 JSONL files."
    )
    parser.add_argument(
        "--s3-root",
        default=DEFAULT_S3_ROOT,
        help=f"S3 root to scan (default: {DEFAULT_S3_ROOT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSONL path, local or s3://...",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--type-key", default="tp", help="Input row type key.")
    parser.add_argument("--ad-type", default="ad", help="Value in --type-key treated as an ad.")
    parser.add_argument("--id-key", default="id", help="Input ID key.")
    parser.add_argument("--lang-key", default="lg", help="Input language key.")
    parser.add_argument("--text-key", default="ft", help="Input full-text key.")
    parser.add_argument(
        "--min-ft-length",
        type=int,
        default=1,
        help="Minimum stripped ft length to keep.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output path.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Log progress every N scanned files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    parser.add_argument("--log-file", help="Optional log file path.")
    return parser.parse_args(argv)


def parse_s3_root(s3_root: str) -> Tuple[str, str]:
    match = re.match(r"^s3://([^/]+)(?:/(.*))?$", s3_root)
    if not match:
        raise ValueError(f"Invalid S3 root: {s3_root}")
    return match.group(1), match.group(2) or ""


def is_supported_file(key: str) -> bool:
    return key.endswith(SUPPORTED_SUFFIXES)


def output_exists(path: str) -> bool:
    if not path.startswith("s3://"):
        return os.path.exists(path)

    match = re.match(r"^s3://([^/]+)/(.+)$", path)
    if not match:
        raise ValueError(f"Invalid S3 output path: {path}")

    s3 = agg.get_s3_client()
    try:
        s3.head_object(Bucket=match.group(1), Key=match.group(2))
        return True
    except s3.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            return False
        raise


def article_group(article_id: str) -> Tuple[str, str]:
    newspaper = extract_newspaper_id(article_id)
    year = extract_year(article_id)
    if not newspaper or not re.fullmatch(r"\d{4}", year):
        raise ValueError(f"Could not parse newspaper/year from id={article_id!r}")
    return newspaper, year


def random_score(article_id: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{article_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def iter_ad_rows(
    path: str,
    transport_params: Dict[str, Any],
    args: argparse.Namespace,
    counters: Counters,
) -> Iterator[Dict[str, Any]]:
    filters = {args.type_key: args.ad_type}
    keys = [args.id_key, args.lang_key, args.text_key]

    with smart_open(path, "r", encoding="utf-8", transport_params=transport_params) as handle:
        for line_no, line in enumerate(handle, start=1):
            counters.lines_read += 1
            stripped = line.strip()
            if not stripped:
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                counters.skipped_invalid_json += 1
                LOG.warning("Skipping invalid JSON in %s:%s", path, line_no)
                continue

            row = agg.process_jsonl_file(
                data=data,
                keys=keys,
                filters=filters,
                jq_filter=None,
            )
            if row:
                counters.ad_rows_seen += 1
                yield row


def sample_rows(args: argparse.Namespace) -> Tuple[Dict[Tuple[str, str], Candidate], Counters]:
    bucket, prefix = parse_s3_root(args.s3_root)
    s3 = agg.get_s3_client()
    transport_params = {"client": s3}
    samples: Dict[Tuple[str, str], Candidate] = {}
    counters = Counters()

    for key in agg.yield_s3_objects(bucket, prefix):
        counters.files_seen += 1
        if not is_supported_file(key):
            continue

        counters.files_scanned += 1
        path = f"s3://{bucket}/{key}"
        LOG.info("Scanning %s", path)

        for row in iter_ad_rows(path, transport_params, args, counters):
            article_id = row.get(args.id_key)
            if not isinstance(article_id, str) or not article_id.strip():
                counters.skipped_invalid_id += 1
                continue

            text = row.get(args.text_key)
            if not isinstance(text, str) or len(text.strip()) < args.min_ft_length:
                counters.skipped_missing_text += 1
                continue

            try:
                group = article_group(article_id)
            except ValueError:
                counters.skipped_invalid_id += 1
                LOG.warning("Skipping malformed article id: %r", article_id)
                continue

            score = random_score(article_id, args.seed)
            output_row = {
                "id": article_id,
                "lg": row.get(args.lang_key),
                "ft": text.strip(),
                "type": "ad",
            }

            current = samples.get(group)
            if current is None or (score, article_id) < (current.score, current.row["id"]):
                samples[group] = Candidate(score=score, row=output_row)

        if args.progress_every and counters.files_scanned % args.progress_every == 0:
            LOG.info(
                "Progress: scanned_files=%s lines_read=%s ad_rows_seen=%s sampled_groups=%s",
                counters.files_scanned,
                counters.lines_read,
                counters.ad_rows_seen,
                len(samples),
            )

    counters.sampled_groups = len(samples)
    return samples, counters


def write_output(rows: Iterator[Dict[str, Any]], output_path: str) -> None:
    suffix = ".jsonl"
    with tempfile.NamedTemporaryFile(
        delete=False,
        mode="w",
        encoding="utf-8",
        suffix=suffix,
    ) as handle:
        tmp_path = handle.name
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        if output_path.startswith("s3://"):
            agg.upload_to_s3(tmp_path, output_path, agg.get_s3_client())
            return

        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        shutil.move(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.min_ft_length < 0:
        raise SystemExit("--min-ft-length must be greater than or equal to 0")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be greater than or equal to 0")
    if output_exists(args.output) and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output}. Use --overwrite to replace it.")

    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(agg.SmartFileHandler(args.log_file, mode="w"))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)-15s %(filename)s:%(lineno)d %(levelname)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    samples, counters = sample_rows(args)
    sorted_rows = (
        samples[group].row
        for group in sorted(samples, key=lambda item: (item[0], item[1]))
    )
    write_output(sorted_rows, args.output)

    LOG.info("Finished.")
    LOG.info("Files seen: %s", counters.files_seen)
    LOG.info("Files scanned: %s", counters.files_scanned)
    LOG.info("Lines read: %s", counters.lines_read)
    LOG.info("Ad rows seen: %s", counters.ad_rows_seen)
    LOG.info("Sampled newspaper-year groups: %s", counters.sampled_groups)
    LOG.info("Skipped invalid JSON rows: %s", counters.skipped_invalid_json)
    LOG.info("Skipped invalid IDs: %s", counters.skipped_invalid_id)
    LOG.info("Skipped missing/short ft: %s", counters.skipped_missing_text)
    LOG.info("Output: %s", args.output)


if __name__ == "__main__":
    main()
