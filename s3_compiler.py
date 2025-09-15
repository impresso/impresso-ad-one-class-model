#!/usr/bin/env python3
"""
S3 Dataset Compiler Tool

This script compiles a corpus by taking a JSONL file with record IDs and fetching
corresponding entries from S3 files based on ID patterns. It extracts content from
matching lines in S3 JSONL.bz2 files and applies optional JQ transformations.

Features:
- Reads a JSONL file containing record IDs
- Parses IDs to determine corresponding S3 files (e.g., NEWSPAPER-YEAR-CONTENTITEMID)
- Fetches matching records from S3 JSONL.bz2 files
- Supports JQ-based transformation of extracted records
- Memory-efficient processing with caching of frequently accessed files
- Outputs compiled corpus to a local file or S3 path

Usage:
    python s3_compiler.py \
        -i sample_ids.jsonl \
        --s3-prefix s3://bucket/path/to/dataset \
        -o compiled_corpus.jsonl \
        --id-field id \
        --log-level INFO

    # With transformation:
    python s3_compiler.py \
        -i sample_ids.jsonl \
        --s3-prefix s3://bucket/path/to/dataset \
        -o compiled_corpus.jsonl \
        --id-field id \
        --transform-expr '{id: .id, title: .title, content: .content_text}' \
        --log-level INFO

Examples:
1. Basic compilation:
    python s3_compiler.py -i ids.jsonl --s3-prefix s3://bucket/data \
        -o corpus.jsonl --id-field id

2. With transformation:
    python s3_compiler.py -i ids.jsonl --s3-prefix s3://bucket/data \
        -o corpus.jsonl --id-field id \
        --transform-expr '{id: .id, text: .content_text, date: .date}'

3. Using transform file:
    python s3_compiler.py -i ids.jsonl --s3-prefix s3://bucket/data \
        -o corpus.jsonl --id-field id --transform-file transform.jq

ID Format:
The script expects IDs in the format: NEWSPAPER-YEAR-CONTENTITEMID
It will search for files ending with: NEWSPAPER-YEAR.jsonl.bz2 in the S3 prefix
Files may be in nested directories and optionally have prefixes
(e.g., path/to/NEWSPAPER-YEAR.jsonl.bz2 or path/to/PREFIX-NEWSPAPER-YEAR.jsonl.bz2)
"""

import argparse
import json
import logging
import re
import sys
import tempfile
from typing import Dict, Any, Optional, List

import jq
from dotenv import load_dotenv
from smart_open import open as smart_open

try:
    from impresso_cookbook import (
        get_s3_client,
        setup_logging,
        get_transport_params,
        parse_s3_path,
    )
except ImportError:
    # Fallback for when impresso_cookbook is not available
    from common import (
        get_s3_client,
        setup_logging,
        get_transport_params,
        parse_s3_path,
    )

log = logging.getLogger(__name__)

# Load environment variables for S3 credentials
load_dotenv()


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        args: Command-line arguments (uses sys.argv if None)

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Compile corpus from S3 by fetching records matching IDs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Logging options
    parser.add_argument(
        "--log-file", dest="log_file", help="Write log to FILE", metavar="FILE"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: %(default)s)",
    )

    # Input/Output options
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Input JSONL file containing record IDs (local or S3)",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        required=True,
        help="S3 path prefix (e.g., s3://bucket/prefix) to read JSONL.bz2 files",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        type=str,
        required=True,
        help="Output path for compiled corpus (local or S3), JSONL format.",
    )

    # ID processing options
    parser.add_argument(
        "--id-field",
        type=str,
        default="id",
        help="Field name containing the ID in input records (default: %(default)s)",
    )
    parser.add_argument(
        "--id-pattern",
        type=str,
        default=r"([^-]+)-(\d{4})-(.+)$",
        help="Regex pattern to parse ID (default: NEWSPAPER-YEAR-CONTENTID format)",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="{newspaper}-{year}.jsonl.bz2",
        help="Pattern for S3 file names (default: %(default)s)",
    )

    # Processing options
    parser.add_argument(
        "--transform-expr",
        type=str,
        help="JQ expression to transform extracted records",
    )
    parser.add_argument(
        "--transform-file",
        type=str,
        help="Path to file containing JQ transform expression",
    )
    parser.add_argument(
        "--match-field",
        type=str,
        default="id",
        help="Field name to match against in S3 records (default: %(default)s)",
    )

    parser.add_argument(
        "--include-from-input",
        type=str,
        nargs="*",
        help="Fields from input file to include in output (e.g., 'score', 'label')",
    )

    return parser.parse_args(args)


class S3CompilerProcessor:
    """
    A processor class that compiles a corpus by fetching records from S3
    based on provided IDs.
    """

    def __init__(
        self,
        input_file: str,
        s3_prefix: str,
        output_file: str,
        id_field: str = "id",
        id_pattern: str = r"([^-]+)-(\d{4})-(.+)$",
        file_pattern: str = "{newspaper}-{year}.jsonl.bz2",
        transform_expr: Optional[str] = None,
        transform_file: Optional[str] = None,
        match_field: str = "id",
        include_from_input: Optional[List[str]] = None,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
    ) -> None:
        """
        Initialize the S3CompilerProcessor.

        Args:
            input_file: Path to input JSONL file with IDs
            s3_prefix: S3 prefix path to read from
            output_file: Path to the output file
            id_field: Field name containing ID in input records
            id_pattern: Regex pattern to parse IDs
            file_pattern: Pattern for S3 file names
            transform_expr: JQ expression for transformation
            transform_file: Path to JQ transform file
            match_field: Field name to match in S3 records
            include_from_input: Fields from input file to include in output
            log_level: Logging level
            log_file: Path to log file
        """
        self.input_file = input_file
        self.s3_prefix = s3_prefix
        self.output_file = output_file
        self.id_field = id_field
        self.id_pattern = id_pattern
        self.file_pattern = file_pattern
        self.transform_expr = transform_expr
        self.transform_file = transform_file
        self.match_field = match_field
        self.include_from_input = include_from_input or []
        self.log_level = log_level
        self.log_file = log_file

        # Configure logging
        setup_logging(self.log_level, self.log_file, logger=log)

        # Initialize S3 client
        self.s3_client = get_s3_client()

        # Compile regex pattern
        try:
            self.id_regex = re.compile(self.id_pattern)
        except re.error as e:
            log.error(f"Invalid ID pattern '{self.id_pattern}': {e}")
            sys.exit(1)

        # Compile JQ expression
        self._compile_jq_expressions()

        # Initialize file cache and statistics
        self.file_key_cache: Dict[str, Optional[str]] = {}  # Cache for file lookups
        # Cache for input record info
        self.id_to_record_info: Dict[str, Dict[str, Any]] = {}
        self.statistics = {
            "input_records": 0,
            "parsed_ids": 0,
            "found_records": 0,
            "files_loaded": 0,
        }

    def _compile_jq_expressions(self) -> None:
        """Compile JQ transformation expression."""
        self.transform_program = None
        if self.transform_expr or self.transform_file:
            transform_code = self.transform_expr
            if self.transform_file:
                try:
                    with smart_open(
                        self.transform_file,
                        "r",
                        encoding="utf-8",
                        transport_params=get_transport_params(self.transform_file),
                    ) as f:
                        transform_code = f.read().strip()
                except Exception as e:
                    log.error(
                        f"Failed to load transform file '{self.transform_file}': {e}"
                    )
                    sys.exit(1)

            try:
                self.transform_program = jq.compile(transform_code)
                log.info("Compiled transform JQ expression")
            except Exception as e:
                log.error(f"Invalid transform JQ expression: {e}")
                sys.exit(1)

    def _parse_id(self, record_id: str) -> Optional[Dict[str, str]]:
        """
        Parse an ID to extract components for file lookup.

        Args:
            record_id: The record ID to parse

        Returns:
            Optional[Dict[str, str]]: Parsed components or None if parsing fails
        """
        match = self.id_regex.match(record_id)
        if not match:
            log.debug(f"ID '{record_id}' does not match pattern '{self.id_pattern}'")
            return None

        groups = match.groups()
        if len(groups) >= 2:
            return {
                "newspaper": groups[0],
                "year": groups[1],
                "content_id": groups[2] if len(groups) > 2 else "",
                "full_id": record_id,
            }
        return None

    def _get_file_key(self, parsed_id: Dict[str, str]) -> Optional[str]:
        """
        Find S3 file key from parsed ID components by searching for matching files.
        Uses caching to avoid repeated S3 list operations for the same file pattern.

        Args:
            parsed_id: Parsed ID components

        Returns:
            Optional[str]: S3 file key if found, None otherwise
        """
        # Create cache key from newspaper and year
        cache_key = f"{parsed_id['newspaper']}-{parsed_id['year']}"

        # Check cache first
        if cache_key in self.file_key_cache:
            cached_result = self.file_key_cache[cache_key]
            if cached_result is not None:
                log.debug(f"Using cached file key: {cached_result}")
            return cached_result

        bucket, prefix = parse_s3_path(self.s3_prefix)

        # Search pattern: files ending with NEWSPAPER-YEAR.jsonl.bz2
        # Files may have optional prefixes or be in nested directories
        search_suffix = f"{parsed_id['newspaper']}-{parsed_id['year']}.jsonl.bz2"

        try:
            # List objects with the prefix to find matching files
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=bucket, Prefix=prefix.rstrip("/") + "/" if prefix else ""
            )

            for page in page_iterator:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    # Check if the key ends with our target pattern
                    if key.endswith(search_suffix):
                        log.debug(f"Found matching file: {key}")
                        # Cache the result
                        self.file_key_cache[cache_key] = key
                        return key

        except Exception as e:
            log.debug(f"Error searching for file with pattern {search_suffix}: {e}")

        log.debug(f"No file found matching pattern {search_suffix}")
        # Cache the negative result
        self.file_key_cache[cache_key] = None
        return None

    def _process_s3_file_streaming(
        self, bucket: str, file_key: str, target_ids: List[str], outfile
    ) -> int:
        """
        Stream through an S3 file and process only the records we need.
        This avoids loading the entire file into memory.
        Stops processing when all target IDs have been found.

        Args:
            bucket: S3 bucket name
            file_key: S3 file key
            target_ids: List of IDs to look for in this file
            outfile: Output file handle to write results to

        Returns:
            int: Number of records found and processed
        """
        # Convert to set for O(1) lookup and track remaining IDs
        remaining_ids = set(target_ids)
        records_found = 0
        transport_params = get_transport_params(f"s3://{bucket}/{file_key}")

        try:
            with smart_open(
                f"s3://{bucket}/{file_key}",
                "rb",
                transport_params=transport_params,
            ) as infile:
                for line_num, line in enumerate(infile, 1):
                    try:
                        record = json.loads(line)
                        if self.match_field in record:
                            match_value = str(record[self.match_field])
                            # Only process if this ID is in our remaining target list
                            if match_value in remaining_ids:
                                records_found += 1
                                self.statistics["found_records"] += 1

                                # Remove from remaining IDs
                                remaining_ids.remove(match_value)

                                # Apply transformation
                                transformed_record = self._apply_transform(record)
                                if transformed_record is not None:
                                    # Include original ID in output if not already there
                                    if self.id_field not in transformed_record:
                                        transformed_record[self.id_field] = match_value

                                    # Include fields from input file
                                    # We need to get the record_info for this ID
                                    record_info = self._get_record_info_for_id(
                                        match_value
                                    )
                                    if record_info:
                                        for field in self.include_from_input:
                                            if field in record_info:
                                                transformed_record[field] = record_info[
                                                    field
                                                ]

                                    outfile.write(
                                        json.dumps(
                                            transformed_record, ensure_ascii=False
                                        )
                                        + "\n"
                                    )

                                # Early exit if all target IDs have been found
                                if not remaining_ids:
                                    log.debug(
                                        f"All {len(target_ids)} target IDs found, "
                                        f"stopping processing of {file_key} at line "
                                        f"{line_num}"
                                    )
                                    break

                    except json.JSONDecodeError:
                        log.debug(f"Skipping malformed line {line_num} in {file_key}")
                    except Exception as e:
                        log.debug(
                            f"Error processing line {line_num} in {file_key}: {e}"
                        )

            if remaining_ids:
                missing_sample = list(remaining_ids)[:5]
                suffix = "..." if len(remaining_ids) > 5 else ""
                log.debug(
                    f"{len(remaining_ids)} target IDs not found in {file_key}: "
                    f"{missing_sample}{suffix}"
                )

            log.info(
                f"Found {records_found}/{len(target_ids)} target records in {file_key}"
            )
            self.statistics["files_loaded"] += 1

        except Exception as e:
            log.warning(f"Failed to process file {file_key}: {e}")

        return records_found

    def _get_record_info_for_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the cached record info for a given ID.

        Args:
            record_id: The record ID to look up

        Returns:
            Optional[Dict[str, Any]]: Record info if found, None otherwise
        """
        return self.id_to_record_info.get(record_id)

    def _group_ids_by_pattern(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Read input file and group IDs by their newspaper-year pattern.
        This avoids S3 operations during the grouping phase.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Mapping from newspaper-year patterns to
            list of records containing ID and input data
        """
        pattern_to_records: Dict[str, List[Dict[str, Any]]] = {}

        with smart_open(
            self.input_file,
            "r",
            encoding="utf-8",
            transport_params=get_transport_params(self.input_file),
        ) as infile:
            for line_num, line in enumerate(infile, 1):
                try:
                    input_record = json.loads(line)
                    self.statistics["input_records"] += 1

                    if self.id_field not in input_record:
                        log.warning(f"Line {line_num}: Missing '{self.id_field}' field")
                        continue

                    record_id = str(input_record[self.id_field])
                    parsed_id = self._parse_id(record_id)

                    if parsed_id is None:
                        continue

                    self.statistics["parsed_ids"] += 1

                    # Group by newspaper-year pattern (not actual file key yet)
                    pattern_key = f"{parsed_id['newspaper']}-{parsed_id['year']}"

                    # Store both ID and selected input fields
                    record_info = {"id": record_id}
                    for field in self.include_from_input:
                        if field in input_record:
                            record_info[field] = input_record[field]

                    # Cache the record info for later lookup
                    self.id_to_record_info[record_id] = record_info

                    if pattern_key not in pattern_to_records:
                        pattern_to_records[pattern_key] = []
                    pattern_to_records[pattern_key].append(record_info)

                except json.JSONDecodeError:
                    log.warning(f"Skipping malformed line {line_num}")
                except Exception as e:
                    log.error(f"Error processing line {line_num}: {e}")

        log.info(
            f"Grouped {self.statistics['parsed_ids']} IDs into "
            f"{len(pattern_to_records)} newspaper-year patterns"
        )
        return pattern_to_records

    def _apply_transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply transformation to a record.

        Args:
            record: JSON record to transform

        Returns:
            Optional[Dict[str, Any]]: Transformed record or None if transformation fails
        """
        if self.transform_program is None:
            return record

        try:
            return self.transform_program.input(record).first()
        except StopIteration:
            log.debug("Transform returned empty result")
            return None
        except Exception as e:
            log.debug(f"Transform error for record: {e}")
            return None

    def _upload_to_s3(self, local_path: str, s3_path: str) -> None:
        """Upload a local file to S3."""
        bucket, key = parse_s3_path(s3_path)
        self.s3_client.upload_file(local_path, bucket, key)
        log.info(f"Uploaded {local_path} to {s3_path}")

    def run(self) -> None:
        """Run the compilation process."""
        try:
            log.info(f"Starting compilation from {self.input_file}")
            log.info(f"Looking up records in {self.s3_prefix}")

            # Group IDs by newspaper-year patterns for efficient processing
            pattern_to_records = self._group_ids_by_pattern()

            # Create temporary file for output
            suffix = self.output_file.split(".")[-1]
            with tempfile.NamedTemporaryFile(
                delete=False, mode="w", encoding="utf-8", suffix=f".{suffix}"
            ) as tmpfile:
                tmpfile_path = tmpfile.name
                log.info(f"Temporary file created: {tmpfile_path}")

                bucket, _ = parse_s3_path(self.s3_prefix)

                with smart_open(tmpfile_path, "w", encoding="utf-8") as outfile:
                    # Process each newspaper-year pattern
                    for pattern_key, record_list in pattern_to_records.items():
                        log.info(
                            f"Processing pattern {pattern_key} with "
                            f"{len(record_list)} IDs"
                        )

                        # Parse the pattern to get newspaper and year
                        newspaper, year = pattern_key.split("-", 1)
                        parsed_id = {"newspaper": newspaper, "year": year}

                        # Find the S3 file for this pattern
                        file_key = self._get_file_key(parsed_id)

                        if file_key is None:
                            log.warning(f"No S3 file found for pattern {pattern_key}")
                            continue

                        log.info(f"Found S3 file: s3://{bucket}/{file_key}")

                        # Check if file exists in S3
                        try:
                            self.s3_client.head_object(Bucket=bucket, Key=file_key)
                        except Exception:
                            log.warning(f"File {file_key} not found in S3")
                            continue

                        # Extract just the IDs for this pattern
                        target_ids = [record["id"] for record in record_list]

                        # Stream through the S3 file and process matching records
                        self._process_s3_file_streaming(
                            bucket, file_key, target_ids, outfile
                        )

                # Upload to S3 or move to final location
                if self.output_file.startswith("s3://"):
                    self._upload_to_s3(tmpfile_path, self.output_file)
                else:
                    import shutil

                    shutil.move(tmpfile_path, self.output_file)
                    log.info(f"Moved {tmpfile_path} to {self.output_file}")

            # Log summary statistics
            log.info("Compilation complete:")
            log.info(f"  Input records processed: {self.statistics['input_records']}")
            log.info(f"  IDs successfully parsed: {self.statistics['parsed_ids']}")
            log.info(f"  Records found in S3: {self.statistics['found_records']}")
            log.info(f"  Files loaded from S3: {self.statistics['files_loaded']}")
            log.info(f"  Unique patterns processed: {len(pattern_to_records)}")

            if self.statistics["input_records"] > 0:
                success_rate = (
                    self.statistics["found_records"] / self.statistics["input_records"]
                )
                log.info(f"  Success rate: {success_rate:.4f}")

            log.info(f"Results saved to {self.output_file}")

        except Exception as e:
            log.error(f"Error during compilation process: {e}", exc_info=True)
            sys.exit(1)


def main(args: Optional[List[str]] = None) -> None:
    """
    Main function to run the S3 Compiler Processor.

    Args:
        args: Command-line arguments (uses sys.argv if None)
    """
    options: argparse.Namespace = parse_arguments(args)

    processor: S3CompilerProcessor = S3CompilerProcessor(
        input_file=options.input_file,
        s3_prefix=options.s3_prefix,
        output_file=options.output_file,
        id_field=options.id_field,
        id_pattern=options.id_pattern,
        file_pattern=options.file_pattern,
        transform_expr=options.transform_expr,
        transform_file=options.transform_file,
        match_field=options.match_field,
        include_from_input=options.include_from_input,
        log_level=options.log_level,
        log_file=options.log_file,
    )

    # Log the parsed options after logger is configured
    log.info("%s", options)

    processor.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"Processing error: {e}", exc_info=True)
        sys.exit(2)
