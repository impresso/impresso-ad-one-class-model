#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
import tempfile
from typing import Iterator, List, Optional, Sequence

'''
python helper_scripts_for_data_preparation/s3_multi_prefix_runner.py \
  --s3-root s3://42-processed-data-final/topics/topics-tm-mallet_infer_seed42_v2.0.1-multilingual_v2-0-1/DFS \
  --output non_ad_articles.jsonl \
  --jq-filter helper_scripts_for_data_preparation/topics_non_ads_filter.jq \
  -k ci_id lg
'''

# Ensure we can import the sibling module s3_aggregator.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
# cookbook helpers (common.py, s3_aggregator.py) live in the cookbook submodule
COOKBOOK_LIB_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "cookbook", "lib")
if COOKBOOK_LIB_DIR not in sys.path:
    sys.path.append(COOKBOOK_LIB_DIR)

import jq  # noqa: E402
from smart_open import open as smart_open  # noqa: E402

# Import functions from the existing aggregator
import s3_aggregator as agg  # noqa: E402


def list_immediate_subprefixes(bucket: str, root_prefix: str) -> Iterator[str]:
    """
    Yield immediate child "folders" (sub-prefixes) under root_prefix in the given bucket.
    """
    s3 = agg.get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    # Normalize: empty string for bucket root; otherwise ensure trailing slash
    if root_prefix:
        prefix = root_prefix if root_prefix.endswith("/") else f"{root_prefix}/"
    else:
        prefix = ""

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            yield cp["Prefix"]


def parse_arguments(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run s3_aggregator over each immediate subfolder of a root S3 prefix and aggregate into a single output.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--s3-root",
        type=str,
        required=True,
        help="Root S3 path containing multiple folders (s3://BUCKET or s3://BUCKET/ROOT_PREFIX)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the final aggregated output JSONL (local or S3). Example: s3://your-bucket/output.jsonl",
    )
    parser.add_argument(
        "-k",
        "--keys",
        type=str,
        nargs="+",
        help="List of keys to extract from each JSON object (passed to s3_aggregator)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="+",
        help="List of filters in the form FEAT=VAL (passed to s3_aggregator)",
    )
    parser.add_argument(
        "--jq-filter",
        type=str,
        help="Path to a jq filter file to apply to each JSON object (passed to s3_aggregator)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument("--log-file", type=str, help="Log file path")
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=".",
        help="Directory for temporary files (default: current directory)",
    )
    return parser.parse_args(args)





def main(cli_args: Optional[Sequence[str]] = None) -> None:
    if cli_args is None:
        cli_args = sys.argv[1:]
    options = parse_arguments(cli_args)

    handlers = [logging.StreamHandler()]
    if options.log_file:
        handlers.append(agg.SmartFileHandler(options.log_file, mode="w"))
    logging.basicConfig(
        level=options.log_level,
        format="%(asctime)-15s %(filename)s:%(lineno)d %(levelname)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.info(f"Arguments: {options}")

    # Ensure tmp dir exists
    tmp_dir = os.path.abspath(options.tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    logging.info(f"Using tmp dir: {tmp_dir}")

    # Parse and validate root S3 (allow bucket-only)
    m = re.match(r"^s3://([^/]+)(?:/(.*))?$", options.s3_root)
    if not m:
        logging.error("Invalid --s3-root format. Expected s3://BUCKET or s3://BUCKET/PREFIX")
        sys.exit(1)
    bucket, root_prefix = m.group(1), (m.group(2) or "")

    # Validate output if S3
    if options.output.startswith("s3://"):
        out_match = re.match(r"s3://([^/]+)/(.+)", options.output)
        if not out_match:
            logging.error("Invalid --output format for S3. Expected s3://BUCKET/KEY")
            sys.exit(1)

    # Prepare filters and jq filter
    filters = agg.parse_filter_arguments(options.filter) if options.filter else {}
    jq_filter = None
    if options.jq_filter:
        with open(options.jq_filter, "r", encoding="utf-8") as jq_file:
            jq_content = jq_file.read()
            logging.info(f"Loaded jq filter from {options.jq_filter}")
            try:
                jq_filter = jq.compile(jq_content)
                logging.info("JQ filter compiled successfully")
            except Exception as e:
                logging.error(f"Failed to compile jq filter: {e}")
                sys.exit(1)

    # Discover subfolders or process the provided prefix directly
    if root_prefix:
        # User provided a concrete folder like DTT/; no need to go deeper
        logging.info("Non-empty root prefix provided; skipping discovery and processing it directly.")
        subprefixes: List[str] = [root_prefix if root_prefix.endswith("/") else f"{root_prefix}/"]
    else:
        # List only immediate subfolders under the bucket root
        subprefixes = list(list_immediate_subprefixes(bucket, root_prefix))
        # Enforce "first-level only" (e.g., 'ACI/' -> 1 slash; 'DTT/2020/' -> 2 slashes -> excluded)
        subprefixes = [sp for sp in subprefixes if sp.count("/") == 1]
        if not subprefixes:
            logging.warning("No subfolders found; nothing to process at the bucket root.")
    logging.info(f"Discovered {len(subprefixes)} top-level folder(s): {subprefixes}")

    # Decide aggregate target: local writes directly; S3 uses temp then upload
    is_s3_output = options.output.startswith("s3://")
    suffix = options.output.split(".")[-1] if "." in options.output else "jsonl"
    final_tmp_path = None
    if is_s3_output:
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", encoding="utf-8", suffix=f".{suffix}", dir=tmp_dir
        ) as final_tmp:
            final_tmp_path = final_tmp.name
        aggregate_target_path = final_tmp_path
        logging.info(f"Final temporary aggregate file (S3 mode): {aggregate_target_path}")
    else:
        aggregate_target_path = os.path.abspath(options.output)
        # Truncate/create file so it's visible and empty before appending
        with open(aggregate_target_path, "w", encoding="utf-8"):
            pass
        logging.info(f"Aggregating directly into: {aggregate_target_path}")

    total_aggregated_lines = 0

    try:
        # Process each subfolder separately via s3_aggregator, append to aggregate target
        for sp in subprefixes:
            display_uri = f"s3://{bucket}/{sp}" if sp else f"s3://{bucket}/"
            logging.info(f"Processing subfolder: {display_uri}")

            # Temporary output for this subfolder
            with tempfile.NamedTemporaryFile(
                delete=False, mode="w", encoding="utf-8", suffix=f".{suffix}", dir=tmp_dir
            ) as part_tmp:
                part_output_path = part_tmp.name

            try:
                # Run the existing aggregator for this subfolder into a local temp file
                agg.process_s3_files(
                    bucket=bucket,
                    prefix=sp,
                    keys=options.keys,
                    filters=filters,
                    output_path=part_output_path,
                    jq_filter=jq_filter,
                )

                # Append its content to the aggregate
                with smart_open(part_output_path, "r", encoding="utf-8") as r, smart_open(
                    aggregate_target_path, "a", encoding="utf-8"
                ) as w:
                    for line in r:
                        w.write(line)
                        total_aggregated_lines += 1
            except Exception as e:
                logging.error(f"Error processing subfolder {display_uri}: {e}")
                # Continue with next subfolder instead of failing completely
                continue
            finally:
                # Cleanup partial file
                try:
                    os.remove(part_output_path)
                except OSError:
                    pass

        # Finalize output
        if is_s3_output:
            s3 = agg.get_s3_client()
            agg.upload_to_s3(aggregate_target_path, options.output, s3)
            logging.info(f"Uploaded aggregate to {options.output}")
        else:
            logging.info(f"Wrote aggregate to {aggregate_target_path}")

        logging.info(f"Total lines aggregated: {total_aggregated_lines}")
    finally:
        # Cleanup final temp only if we used a temp (S3 mode)
        if is_s3_output and final_tmp_path:
            try:
                if os.path.exists(final_tmp_path):
                    os.remove(final_tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
