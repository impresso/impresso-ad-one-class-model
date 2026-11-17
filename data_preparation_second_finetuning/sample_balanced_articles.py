#!/usr/bin/env python3
"""Sample balanced ad/non-ad article ids by newspaper and year.

This script scans JSONL-style files under a local directory or S3 prefix and
randomly samples a fixed number of article rows per class for each
`newspaper_id + year` group, using article ids to derive the group.

Expected input row shape:
{"id": "GDL-1985-10-28-a-i0001", "tp": "ar", "ad_classification": "ad"}

Only rows with `tp == "ar"` are considered by default.

Output rows contain:
{"id": "...", "ad_classification": "ad" | "non-ad"}

When --text-root is set, matching rebuilt text is added as well:
{"id": "...", "ad_classification": "ad" | "non-ad", "ft": "..."}

Examples:
python data_preparation_second_finetuning/sample_balanced_articles.py \
  --input-root s3://140-processed-data-sandbox/content-item-classification/content-item-classification-base-multilingual_v1-0-0/ \
  --output sampled.jsonl

python data_preparation_second_finetuning/sample_balanced_articles.py \
  --input-root s3://140-processed-data-sandbox/content-item-classification/content-item-classification-base-multilingual_v1-0-0/ \
  --text-root s3://122-rebuilt-final/ \
  --output ads-nonads_inspection_sample_2.jsonl \
  --per-class 2 \
  --seed 42
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv
from smart_open import open as smart_open

# Allow running the script directly without installing the package.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
# cookbook helpers (common.py, s3_aggregator.py) live in the cookbook submodule
COOKBOOK_LIB_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "cookbook", "lib")
if COOKBOOK_LIB_DIR not in sys.path:
    sys.path.append(COOKBOOK_LIB_DIR)

from common import (  # noqa: E402
    extract_newspaper_id,
    extract_year,
    get_transport_params,
    parse_s3_path,
    setup_logging,
    upload_file_to_s3,
    yield_s3_objects,
)

load_dotenv()

LOG = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".jsonl", ".jsonl.bz2", ".jsonl.gz")
TEXT_SUFFIX_PREFERENCE = (".jsonl.bz2", ".jsonl.gz", ".jsonl")


@dataclass
class Reservoir:
    seen: int = 0
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)

    def consider(self, row: Dict[str, Any], k: int, rng: random.Random) -> None:
        """Reservoir-sample up to k rows uniformly from a stream."""
        self.seen += 1
        if len(self.sample_rows) < k:
            self.sample_rows.append(row)
            return

        slot = rng.randint(1, self.seen)
        if slot <= k:
            self.sample_rows[slot - 1] = row


@dataclass
class Counters:
    files_scanned: int = 0
    rows_read: int = 0
    article_rows_seen: int = 0
    classified_rows_seen: int = 0


@dataclass
class TextCounters:
    indexed_files: int = 0
    groups_looked_up: int = 0
    files_opened: int = 0
    rows_read: int = 0
    matched_ids: int = 0
    eligible_texts: int = 0
    missing_files: int = 0
    missing_ft: int = 0
    too_short_ft: int = 0
    too_long_ft: int = 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample balanced ad/non-ad article ids per newspaper-year."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Local file/dir or S3 root like s3://bucket/prefix",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path, local or s3://bucket/key",
    )
    parser.add_argument(
        "--text-root",
        help=(
            "Optional local dir/file or S3 root for article texts "
            "(e.g. s3://122-rebuilt-final/). When set, rows are only eligible "
            "if matching text is found there."
        ),
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=1,
        help="How many ads and non-ads to sample per newspaper-year (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--id-key",
        default="id",
        help="Input key containing the article id (default: id)",
    )
    parser.add_argument(
        "--output-id-key",
        default="id",
        help="Output key for the sampled article id (default: id)",
    )
    parser.add_argument(
        "--type-key",
        default="tp",
        help="Input key containing the row type (default: tp)",
    )
    parser.add_argument(
        "--article-type-value",
        default="ar",
        help="Only rows with this type are eligible (default: ar)",
    )
    parser.add_argument(
        "--label-key",
        default="ad_classification",
        help="Input key containing the ad label (default: ad_classification)",
    )
    parser.add_argument(
        "--output-label-key",
        default="ad_classification",
        help="Output key for the label (default: ad_classification)",
    )
    parser.add_argument(
        "--text-id-key",
        default="id",
        help="Text-source key containing the article id (default: id)",
    )
    parser.add_argument(
        "--text-key",
        default="ft",
        help="Text-source key containing the full text (default: ft)",
    )
    parser.add_argument(
        "--output-text-key",
        default="ft",
        help="Output key for the fetched text (default: ft)",
    )
    parser.add_argument(
        "--min-ft-length",
        type=int,
        default=1,
        help="Minimum stripped text length to keep when --text-root is set (default: 1)",
    )
    parser.add_argument(
        "--max-ft-length",
        type=int,
        default=400,
        help="Maximum stripped text length to keep when --text-root is set (default: 400)",
    )
    parser.add_argument(
        "--ad-value",
        default="ad",
        help="Value treated as ad in the input/output (default: ad)",
    )
    parser.add_argument(
        "--non-ad-value",
        default="non-ad",
        help="Value treated as non-ad in the input/output (default: non-ad)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any newspaper-year has fewer than --per-class items in either class",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument("--log-file", help="Optional local or S3 log path")
    return parser.parse_args(argv)


def normalize_value(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def normalize_label(raw_value: object, ad_value: str, non_ad_value: str) -> Optional[str]:
    """Map input values to internal labels ad/non-ad."""
    if raw_value is None:
        return None

    value = normalize_value(str(raw_value))
    ad_norm = normalize_value(ad_value)
    non_ad_norm = normalize_value(non_ad_value)

    if value == ad_norm:
        return "ad"
    if value in {non_ad_norm, "nonad"}:
        return "non-ad"
    return None


def is_supported_input(path: str) -> bool:
    return path.endswith(SUPPORTED_SUFFIXES)


def iter_local_files(root: str) -> Iterator[str]:
    if os.path.isfile(root):
        if is_supported_input(root):
            yield root
        return

    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            full_path = os.path.join(dirpath, filename)
            if is_supported_input(full_path):
                yield full_path


def parse_s3_root(s3_root: str) -> Tuple[str, str]:
    """Accept s3://bucket or s3://bucket/prefix."""
    if re.fullmatch(r"s3://[^/]+", s3_root):
        bucket = s3_root[5:]
        return bucket, ""
    return parse_s3_path(s3_root)


def iter_input_files(input_root: str) -> Iterator[str]:
    if input_root.startswith("s3://"):
        bucket, prefix = parse_s3_root(input_root)
        for key in yield_s3_objects(bucket, prefix):
            if is_supported_input(key):
                yield f"s3://{bucket}/{key}"
        return

    yield from iter_local_files(input_root)


def iter_json_rows(path: str, counters: Counters) -> Iterator[dict]:
    transport_params = get_transport_params(path)
    with smart_open(path, "r", encoding="utf-8", transport_params=transport_params) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            counters.rows_read += 1
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                LOG.warning("Skipping invalid JSON in %s:%s: %s", path, line_no, exc)


class TextFetcher:
    """Resolve rebuilt files once and stream only the rows needed for sampled IDs."""

    def __init__(
        self,
        text_root: str,
        text_id_key: str,
        text_key: str,
        min_ft_length: int,
        max_ft_length: int,
    ) -> None:
        self.text_root = text_root
        self.text_id_key = text_id_key
        self.text_key = text_key
        self.min_ft_length = min_ft_length
        self.max_ft_length = max_ft_length
        self.stats = TextCounters()
        self.available_paths = list(iter_input_files(text_root))
        self.stats.indexed_files = len(self.available_paths)
        self.group_to_path: Dict[str, Optional[str]] = {}

    def resolve_group_path(self, newspaper_id: str, year: str) -> Optional[str]:
        group_key = f"{newspaper_id}-{year}"
        if group_key in self.group_to_path:
            return self.group_to_path[group_key]

        self.stats.groups_looked_up += 1
        for suffix in TEXT_SUFFIX_PREFERENCE:
            expected_suffix = f"{group_key}{suffix}"
            for path in self.available_paths:
                if path.endswith(expected_suffix):
                    self.group_to_path[group_key] = path
                    return path

        self.group_to_path[group_key] = None
        self.stats.missing_files += 1
        return None

    def fetch_texts(
        self, newspaper_id: str, year: str, target_ids: Set[str]
    ) -> Dict[str, str]:
        if not target_ids:
            return {}

        path = self.resolve_group_path(newspaper_id, year)
        if path is None:
            LOG.warning(
                "No text file found for %s-%s under %s",
                newspaper_id,
                year,
                self.text_root,
            )
            return {}

        texts: Dict[str, str] = {}
        remaining_ids = set(target_ids)
        self.stats.files_opened += 1
        transport_params = get_transport_params(path)

        with smart_open(path, "r", encoding="utf-8", transport_params=transport_params) as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                self.stats.rows_read += 1

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOG.warning("Skipping invalid JSON in %s:%s: %s", path, line_no, exc)
                    continue

                article_id = row.get(self.text_id_key)
                if not isinstance(article_id, str) or article_id not in remaining_ids:
                    continue

                self.stats.matched_ids += 1
                remaining_ids.remove(article_id)

                text = row.get(self.text_key)
                if not isinstance(text, str):
                    self.stats.missing_ft += 1
                else:
                    stripped_text = text.strip()
                    if not stripped_text:
                        self.stats.missing_ft += 1
                    else:
                        text_length = len(stripped_text)
                        if text_length < self.min_ft_length:
                            self.stats.too_short_ft += 1
                        elif text_length > self.max_ft_length:
                            self.stats.too_long_ft += 1
                        else:
                            texts[article_id] = stripped_text
                            self.stats.eligible_texts += 1

                if not remaining_ids:
                    break

        return texts


def build_output_row(
    args: argparse.Namespace,
    article_id: str,
    label: str,
    text_value: Optional[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        args.output_id_key: article_id,
        args.output_label_key: args.ad_value if label == "ad" else args.non_ad_value,
    }
    if text_value is not None:
        row[args.output_text_key] = text_value
    return row


def build_samples(
    args: argparse.Namespace,
) -> Tuple[List[dict], List[str], Counters, Optional[TextCounters]]:
    rng = random.Random(args.seed)
    counters = Counters()
    text_fetcher = (
        TextFetcher(
            text_root=args.text_root,
            text_id_key=args.text_id_key,
            text_key=args.text_key,
            min_ft_length=args.min_ft_length,
            max_ft_length=args.max_ft_length,
        )
        if args.text_root
        else None
    )
    groups: Dict[Tuple[str, str], Dict[str, Reservoir]] = {}

    for path in iter_input_files(args.input_root):
        counters.files_scanned += 1
        LOG.info("Scanning %s", path)
        candidates_by_group: Dict[Tuple[str, str], Dict[str, List[str]]] = {}

        for row in iter_json_rows(path, counters):
            if row.get(args.type_key) != args.article_type_value:
                continue
            counters.article_rows_seen += 1

            label = normalize_label(row.get(args.label_key), args.ad_value, args.non_ad_value)
            if label is None:
                continue
            counters.classified_rows_seen += 1

            article_id = row.get(args.id_key)
            if not isinstance(article_id, str) or not article_id.strip():
                continue

            try:
                newspaper_id = extract_newspaper_id(article_id)
                year = extract_year(article_id)
            except Exception:
                LOG.warning("Skipping malformed article id: %r", article_id)
                continue

            group_key = (newspaper_id, year)
            if group_key not in candidates_by_group:
                candidates_by_group[group_key] = {"ad": [], "non-ad": []}

            candidates_by_group[group_key][label].append(article_id)

        for group_key, label_to_ids in candidates_by_group.items():
            newspaper_id, year = group_key
            if group_key not in groups:
                groups[group_key] = {"ad": Reservoir(), "non-ad": Reservoir()}

            if text_fetcher is None:
                eligible_texts: Dict[str, str] = {}
            else:
                target_ids = set(label_to_ids["ad"]) | set(label_to_ids["non-ad"])
                eligible_texts = text_fetcher.fetch_texts(
                    newspaper_id=newspaper_id,
                    year=year,
                    target_ids=target_ids,
                )

            for label, article_ids in label_to_ids.items():
                for article_id in article_ids:
                    text_value = eligible_texts.get(article_id) if text_fetcher else None
                    if text_fetcher and text_value is None:
                        continue
                    groups[group_key][label].consider(
                        build_output_row(args, article_id, label, text_value),
                        args.per_class,
                        rng,
                    )

    output_rows: List[dict] = []
    incomplete_groups: List[str] = []

    for (newspaper_id, year), label_groups in sorted(groups.items()):
        ad_group = label_groups["ad"]
        non_ad_group = label_groups["non-ad"]
        if ad_group.seen < args.per_class or non_ad_group.seen < args.per_class:
            incomplete_groups.append(
                f"{newspaper_id}-{year} (ad={ad_group.seen}, non-ad={non_ad_group.seen})"
            )
            continue

        output_rows.extend(
            sorted(ad_group.sample_rows, key=lambda row: str(row[args.output_id_key]))
        )
        output_rows.extend(
            sorted(non_ad_group.sample_rows, key=lambda row: str(row[args.output_id_key]))
        )

    return output_rows, incomplete_groups, counters, (
        text_fetcher.stats if text_fetcher is not None else None
    )


def write_output(rows: Iterable[dict], output_path: str) -> None:
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", encoding="utf-8", suffix=".jsonl"
    ) as tmp_handle:
        tmp_path = tmp_handle.name
        for row in rows:
            tmp_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        if output_path.startswith("s3://"):
            from common import get_s3_client  # local import to avoid unused-import noise

            upload_file_to_s3(get_s3_client(), tmp_path, output_path, force_overwrite=True)
        else:
            output_dir = os.path.dirname(os.path.abspath(output_path))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            shutil.move(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.per_class <= 0:
        raise SystemExit("--per-class must be greater than 0")
    if args.text_root and args.min_ft_length < 0:
        raise SystemExit("--min-ft-length must be greater than or equal to 0")
    if args.text_root and args.max_ft_length <= 0:
        raise SystemExit("--max-ft-length must be greater than 0")
    if args.text_root and args.min_ft_length > args.max_ft_length:
        raise SystemExit("--min-ft-length cannot be greater than --max-ft-length")

    setup_logging(args.log_level, args.log_file, force=True)

    output_rows, incomplete_groups, counters, text_counters = build_samples(args)

    if args.strict and incomplete_groups:
        preview = "\n".join(incomplete_groups[:20])
        raise SystemExit(
            "Some newspaper-year groups do not have enough samples:\n"
            f"{preview}\n"
            "Rerun without --strict to skip them."
        )

    write_output(output_rows, args.output)

    complete_groups = len(output_rows) // (2 * args.per_class)
    LOG.info("Files scanned: %s", counters.files_scanned)
    LOG.info("Rows read: %s", counters.rows_read)
    LOG.info("Article rows seen (%s=%s): %s", args.type_key, args.article_type_value, counters.article_rows_seen)
    LOG.info("Classified rows seen: %s", counters.classified_rows_seen)
    LOG.info("Complete newspaper-year groups written: %s", complete_groups)
    LOG.info("Output rows written: %s", len(output_rows))
    if text_counters is not None:
        LOG.info("Text files indexed: %s", text_counters.indexed_files)
        LOG.info("Text groups looked up: %s", text_counters.groups_looked_up)
        LOG.info("Text files opened: %s", text_counters.files_opened)
        LOG.info("Text rows read: %s", text_counters.rows_read)
        LOG.info("Text IDs matched: %s", text_counters.matched_ids)
        LOG.info("Eligible texts found: %s", text_counters.eligible_texts)
        LOG.info("Missing text files: %s", text_counters.missing_files)
        LOG.info("Missing/empty %s values: %s", args.text_key, text_counters.missing_ft)
        LOG.info("Too-short %s values: %s", args.text_key, text_counters.too_short_ft)
        LOG.info("Too-long %s values: %s", args.text_key, text_counters.too_long_ft)
    if incomplete_groups:
        LOG.warning("Skipped incomplete groups: %s", len(incomplete_groups))
        for item in incomplete_groups[:20]:
            LOG.warning("Incomplete: %s", item)
        if len(incomplete_groups) > 20:
            LOG.warning("... and %s more incomplete groups", len(incomplete_groups) - 20)


if __name__ == "__main__":
    main()
