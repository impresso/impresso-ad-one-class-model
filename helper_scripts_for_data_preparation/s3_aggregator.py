#!/usr/bin/env python3
# Patched copy of cookbook/lib/s3_aggregator.py: apply_jq_filter returns None when the
# jq filter yields nothing (row filtered out) instead of raising. Used by s3_multi_prefix_runner.py.
"""
This script processes JSONL.bz2 files stored in an S3 bucket. It extracts specified keys and their values from the JSON objects, applies optional filters, and writes the results to an output file (local or S3).

Features:
- Reads JSONL.bz2 files from an S3 bucket using a specified prefix.
- Extracts specific keys from each JSON object.
- Supports filtering JSON objects based on key-value pairs.
- Allows applying advanced transformations using jq filters.
- Outputs the processed data to a local file or an S3 path.

Usage:
    python3 s3_aggregator.py --s3-prefix s3://your-bucket/your-prefix --keys key1,key2 --output s3://your-output-bucket/output-prefix

Examples:
1. Extract specific keys:
    python3 s3_aggregator.py --s3-prefix s3://your-bucket/your-prefix --keys id,tokens --output s3://your-output-bucket/output.jsonl

2. Apply filters:
    python3 s3_aggregator.py --s3-prefix s3://your-bucket/your-prefix --keys id --filter type=article --output output.jsonl

3. Use a jq filter:
    python3 s3_aggregator.py --s3-prefix s3://your-bucket/your-prefix --jq-filter filter.jq --output output.jsonl
"""

import argparse
import json
import logging
import re
import sys
import tempfile
from typing import Dict, Any, Optional, Sequence
from dotenv import load_dotenv

import jq
from impresso_cookbook import get_s3_client, yield_s3_objects
from smart_open import open as smart_open

load_dotenv()


class SmartFileHandler(logging.FileHandler):
    def _open(self) -> any:
        """Open the log file using smart_open and return a file-like object."""
        return smart_open(self.baseFilename, self.mode, encoding="utf-8")


def list_s3_files_with_prefix(bucket: str, prefix: str) -> Sequence[str]:
    """List all files in an S3 bucket with a given prefix.

    Args:
        bucket (str): S3 bucket name.
        prefix (str): Prefix to filter files.

    Returns:
        Sequence[str]: List of file keys.
    """
    files = [
        file for file in yield_s3_objects(bucket, prefix) if file.endswith("jsonl.bz2")
    ]
    logging.warning(f"Found {len(files)} files with prefix {prefix}")
    return files


def parse_filter_arguments(filter_args: Sequence[str]) -> Dict[str, str]:
    """Parse filter arguments into a dictionary.

    Args:
        filter_args (Sequence[str]): List of filter arguments in the form FEAT=VAL.

    Returns:
        Dict[str, str]: Dictionary of feature-value pairs.
    """
    filters = {}
    for arg in filter_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            filters[key] = value
        else:
            logging.warning(f"Invalid filter argument: {arg}. Skipping.")
    return filters


def apply_jq_filter(
    data: Dict[str, Any], jq_filter: Optional[jq.jq]
) -> Optional[Dict[str, Any]]:
    """Apply a jq filter to a JSON object.

    Args:
        data (Dict[str, Any]): A dictionary representing a single JSON object.
        jq_filter (Optional[jq.jq]): A compiled jq filter.

    Returns:
        Optional[Dict[str, Any]]: The filtered JSON object, or None if the filter excludes it.
    """
    if jq_filter:
        try:
            # Get all results from the filter
            results = list(jq_filter.input(data))
            # Return the first result if any, otherwise None (filtered out)
            return results[0] if results else None
        except Exception as e:
            logging.error(f"Error applying jq filter to data: {json.dumps(data)[:200]}... Error: {e}")
            return None
    return data


def process_jsonl_file(
    data: Dict[str, Any],
    keys: Sequence[str],
    filters: Dict[str, str],
    jq_filter: Optional[jq.jq],
) -> Optional[Dict[str, Any]]:
    """Process a single JSON object, apply filters, jq filter, and extract specified keys.

    Args:
        data (Dict[str, Any]): A dictionary representing a single JSON object.
        keys (Sequence[str]): Keys to extract from the JSON object.
        filters (Dict[str, str]): Filters to apply to the JSON object.
        jq_filter (Optional[jq.jq]): A compiled jq filter.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the extracted keys and values, or None if the input is invalid or does not match filters.
    """
    try:
        for key, value in filters.items():
            logging.debug(
                f"Filtering by {data.get(key)} for {value} {data.get(key) == value}"
            )
            if data.get(key) != value:
                return None
        data = apply_jq_filter(data, jq_filter)
        if data is None:
            return None
        # If no keys are specified, return the entire JSON object
        if not keys:
            return data
        return {key: data[key] for key in keys if key in data}
    except KeyError as e:
        logging.warning(f"Missing key in data: {e}. Skipping this entry.")
        return None


def upload_to_s3(local_path: str, s3_path: str, s3_client) -> None:
    """Upload a local file to an S3 path.

    Args:
        local_path (str): Path to the local file.
        s3_path (str): S3 path to upload the file to.
        s3_client: Boto3 S3 client.
    """
    bucket, key = re.match(r"s3://([^/]+)/(.+)", s3_path).groups()
    s3_client.upload_file(local_path, bucket, key)
    logging.info(f"Uploaded {local_path} to {s3_path}")


def process_s3_files(
    bucket: str,
    prefix: str,
    keys: Sequence[str],
    filters: Dict[str, str],
    output_path: str,
    jq_filter: Optional[jq.jq],
) -> None:
    """Process JSONL.bz2 files in an S3 bucket, apply filters, jq filter, extract specified keys, and write to an output file (local or S3).

    Args:
        bucket (str): S3 bucket name.
        prefix (str): Prefix to filter files.
        keys (Sequence[str]): Keys to extract from each JSON object.
        filters (Dict[str, str]): Filters to apply to each JSON object.
        output_path (str): Path to the output file (local or S3).
        jq_filter (Optional[jq.jq]): A compiled jq filter.
    """
    s3 = get_s3_client()
    transport_params = {"client": s3}
    total_lines = 0
    processed_count = 0

    suffix = output_path.split(".")[-1]
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", encoding="utf-8", suffix=f".{suffix}"
    ) as tmpfile:
        tmpfile_path = tmpfile.name
        logging.info(f"Temporary file created: {tmpfile_path}")
        with smart_open(tmpfile_path, "w", encoding="utf-8") as tmpfile:
            for file_key in yield_s3_objects(bucket, prefix):
                if not file_key.endswith("jsonl.bz2"):
                    continue
                logging.info(f"Processing file: {file_key}")
                with smart_open(
                    f"s3://{bucket}/{file_key}",
                    "rb",
                    transport_params=transport_params,
                ) as infile:
                    for line in infile:
                        total_lines += 1
                        try:
                            data = json.loads(line)
                            processed_data = process_jsonl_file(
                                data, keys, filters, jq_filter
                            )
                            if processed_data:
                                tmpfile.write(
                                    json.dumps(processed_data, ensure_ascii=False)
                                    + "\n"
                                )
                                processed_count += 1
                        except json.JSONDecodeError:
                            logging.error(
                                f"Could not decode JSON from line: {line.strip()}"
                            )
                        except Exception as e:
                            logging.error(f"An error occurred: {e}")

    if output_path.startswith("s3://"):
        upload_to_s3(tmpfile_path, output_path, s3)
    else:
        import shutil

        shutil.move(tmpfile_path, output_path)
        logging.info(f"Moved {tmpfile_path} to {output_path}")

    logging.info(f"Total lines read: {total_lines}")
    logging.info(f"Total items processed and written: {processed_count}")


def parse_arguments(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args (Optional[Sequence[str]]): Command-line arguments.

    Returns:
        argparse.Namespace: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Collect specified keys from JSONL.bz2 files in S3.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help=(
            "Path to store the output JSONL file (local or S3). Example:"
            " s3://your-bucket/output-prefix/output.jsonl"
        ),
    )

    parser.add_argument(
        "-k",
        "--keys",
        type=str,
        nargs="+",
        help="List of keys to extract from each JSON object",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument("--log-file", type=str, help="Log file path")
    parser.add_argument(
        "--s3-prefix",
        type=str,
        required=True,
        help="S3 path prefix (s3://BUCKET/PREFIX) to read JSONL.bz2 files",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="+",
        help="List of filters in the form FEAT=VAL to apply to each JSON object",
    )
    parser.add_argument(
        "--jq-filter",
        type=str,
        help="Path to a jq filter file to apply to each JSON object",
    )
    return parser.parse_args(args)


def main(args: Optional[Sequence[str]] = None) -> None:
    """Main function to run the data extraction process.

    Args:
        args (Optional[Sequence[str]]): Command-line arguments.
    """

    if args is None:
        args = sys.argv[1:]
    options = parse_arguments(args)

    handlers = [logging.StreamHandler()]
    if options.log_file:
        handlers.append(SmartFileHandler(options.log_file, mode="w"))
    logging.basicConfig(
        level=options.log_level,
        format="%(asctime)-15s %(filename)s:%(lineno)d %(levelname)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.info(f"Arguments: {options}")
    logging.info(f"Processing S3 prefix: {options.s3_prefix}")
    match = re.match(r"s3://([^/]+)/(.+)", options.s3_prefix)
    if not match:
        logging.error("Invalid S3 prefix format. Expected s3://BUCKET/PREFIX")
        sys.exit(1)
    bucket, prefix = match.groups()

    output_match = re.match(r"s3://([^/]+)/(.+)", options.output)
    if not output_match and options.output.startswith("s3://"):
        logging.error("Invalid S3 output format. Expected s3://BUCKET/PREFIX")
        sys.exit(1)

    keys = options.keys
    filters = parse_filter_arguments(options.filter) if options.filter else {}
    logging.info(f"FILTERS: {filters}")

    jq_filter = None
    if options.jq_filter:
        try:
            with open(options.jq_filter, "r", encoding="utf-8") as jq_file:
                jq_filter = jq.compile(jq_file.read())
        except Exception as e:
            logging.error(f"Failed to load jq filter: {e}")
            sys.exit(1)

    process_s3_files(bucket, prefix, keys, filters, options.output, jq_filter)

    logging.info(f"Processing complete. Results saved to {options.output}")


if __name__ == "__main__":
    main()
