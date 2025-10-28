"""
Cleans and filters a JSONL file by removing unwanted data or keeping specific keys.

Usage:
    python clean_jsonl.py --input-path input.jsonl [--keys key1 key2 key3] [--min-ft-length 15] [--ocr-threshold 0.5]
"""

import json
import os
import argparse
import re
import gc
from collections import defaultdict
from smart_open import open as smart_open
from impresso_cookbook import get_s3_client, yield_s3_objects
from dotenv import load_dotenv

load_dotenv()


def load_ocrqa_scores(prefix: str) -> dict:
    """
    Loads OCRQA scores from S3 for filtering articles.
    Stores only essential values as tuples (lg, ocrqa) to minimize memory usage.
    
    Args:
        prefix (str): S3 prefix to locate OCRQA files.
        
    Returns:
        Dict[str, tuple]: Mapping of article IDs to (language, ocrqa_score) tuples.
    """
    ocrqa_bucket = "42-processed-data-final"
    ocrqa_root = "ocrqa/ocrqa-ocrqa-wp_v1.0.6_v1-0-0"
    
    print(f"Looking for OCRQA files in: s3://{ocrqa_bucket}/{ocrqa_root}")
    
    s3 = get_s3_client()
    transport_params = {"client": s3}
    ci_scores = {}
    
    # Process in smaller batches to control memory usage
    batch_size = 10000
    current_batch = 0

    try:
        for file_key in yield_s3_objects(ocrqa_bucket, ocrqa_root):
            if not file_key.endswith(".bz2"):
                continue
            print(f"Reading OCRQA file: {file_key}")
            
            # Keep binary mode for compressed files but process line-by-line
            with smart_open(
                f"s3://{ocrqa_bucket}/{file_key}",
                "rb",
                transport_params=transport_params
            ) as f:
                for line_num, line in enumerate(f, 1):
                    # Skip empty lines
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                        
                    try:
                        # Parse only the fields we need using minimal JSON parsing
                        if '"ci_id"' not in line:
                            continue
                            
                        obj = json.loads(line)
                        ci_id = obj.get("ci_id")
                        if ci_id:
                            # Store as tuple (language, ocrqa_score) to save memory
                            ci_scores[ci_id] = (
                                obj.get("lg"),
                                obj.get("ocrqa", 0)
                            )
                        
                        # Periodic cleanup every batch_size lines
                        current_batch += 1
                        if current_batch % batch_size == 0:
                            gc.collect()
                            
                    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError) as e:
                        pass  # Skip malformed lines
                    except Exception as e:
                        print(f"Warning: Failed to parse OCRQA line {line_num}: {e}")
            
            # Force garbage collection after each file
            gc.collect()
    except Exception as e:
        print(f"Error accessing OCRQA files: {e}")
        return {}
    
    print(f"OCRQA loaded for {len(ci_scores)} entries.")
    return ci_scores


def clean_jsonl_file(file_path: str, min_ft_length: int = 15, max_ft_length: int = 400) -> str:
    """
    Cleans a JSONL file and saves the cleaned version with "_cleaned" appended to the original filename.

    Removes records where the 'ft' field is missing, empty, shorter than min_ft_length, or longer than max_ft_length.

    Returns:
        Path to the cleaned JSONL file.
    """
    cleaned_file_path = os.path.splitext(file_path)[0] + "_cleaned.jsonl"

    kept = 0
    dropped = 0
    with open(file_path, 'r', encoding='utf-8') as infile, open(cleaned_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            # Efficient checks first: remove if ft missing/empty/too short/too long
            ft = data.get('ft')
            if not isinstance(ft, str):
                dropped += 1
                continue
            ft_length = len(ft.strip())
            if ft_length < min_ft_length or ft_length > max_ft_length:
                dropped += 1
                continue

            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')
            kept += 1

    print(f"Lines kept: {kept}, dropped: {dropped}")
    return cleaned_file_path


def filter_jsonl(file_path: str, keys_to_keep, min_ft_length: int = 15, max_ft_length: int = 400, ocr_threshold: float = None, s3_prefix: str = None, target_lang: str = None) -> str:
    """
    Filter a JSONL file to keep only specified keys and remove records with short/empty 'ft' or low OCR scores.

    Args:
        file_path: Path to the input JSONL file.
        keys_to_keep: List of keys to retain in each JSON object.
        min_ft_length: Minimum length for the 'ft' field (default 15).
        max_ft_length: Maximum length for the 'ft' field (default 400).
        ocr_threshold: Minimum OCR quality score (default None - no OCR filtering).
        s3_prefix: S3 prefix for loading OCRQA scores (required for OCR filtering).
        target_lang: Target language code to filter by (e.g., 'fr', 'de').

    Returns:
        Path to the filtered JSONL file.
    """
    # Add language code to output filename if specified
    base_name = os.path.splitext(file_path)[0]
    if target_lang:
        filtered_file_path = f"{base_name}_{target_lang}_filtered.jsonl"
    else:
        filtered_file_path = f"{base_name}_filtered.jsonl"

    # Load OCRQA scores only if ocr_threshold is specified
    ocrqa_map = {}
    if ocr_threshold is not None and s3_prefix:
        print(f"Loading OCRQA scores for OCR filtering (threshold: {ocr_threshold})")
        ocrqa_map = load_ocrqa_scores(s3_prefix)

    kept = 0
    dropped = 0
    dropped_ft = 0
    dropped_ocr = 0
    dropped_lang = 0
    
    with open(file_path, 'r', encoding='utf-8') as infile, open(filtered_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            # Language filtering (first check for efficiency)
            if target_lang:
                record_lang = record.get('lg')
                if record_lang != target_lang:
                    dropped += 1
                    dropped_lang += 1
                    continue

            # Skip records with empty, too-short, or too-long 'ft' key (fast check)
            ft = record.get('ft')
            if not isinstance(ft, str):
                dropped += 1
                dropped_ft += 1
                continue
            ft_length = len(ft.strip())
            if ft_length < min_ft_length or ft_length > max_ft_length:
                dropped += 1
                dropped_ft += 1
                continue

            # OCR quality filtering only if threshold is specified and OCRQA data available
            if ocr_threshold is not None and ocrqa_map:
                article_id = record.get('id')
                if article_id and article_id in ocrqa_map:
                    lang, ocr_score = ocrqa_map[article_id]
                    if ocr_score < ocr_threshold:
                        dropped += 1
                        dropped_ocr += 1
                        continue

            # Build filtered record only after passing all checks
            filtered_record = {key: record[key] for key in keys_to_keep if key in record}
            outfile.write(json.dumps(filtered_record, ensure_ascii=False) + '\n')
            kept += 1

    print(f"Filtered file saved to: {filtered_file_path}")
    print(f"Lines kept: {kept}, dropped: {dropped} (ft: {dropped_ft}, ocr: {dropped_ocr}, lang: {dropped_lang})")
    return filtered_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and filter a JSONL file.")
    parser.add_argument("--input-path", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--keys", nargs='*', default=["id", "lg", "ft"], help="List of keys to keep in the output file (default: id, lg, ft).")
    parser.add_argument("--min-ft-length", type=int, default=15, help="Minimum length for the 'ft' field (default: 15).")
    parser.add_argument("--max-ft-length", type=int, default=7100, help="Maximum length for the 'ft' field (default: 400).")
    parser.add_argument("--ocr-threshold", type=float, default=None, help="Minimum OCR quality score (default: None - no OCR filtering).")
    parser.add_argument("--s3-prefix", type=str, default="s3://42-impresso-final/rebuilt_data_rebuilt-wp_v1.0.6_v1-0-0", help="S3 prefix for loading OCRQA scores (default: s3://42-impresso-final/rebuilt_data_rebuilt-wp_v1.0.6_v1-0-0).")
    parser.add_argument("--lang", type=str, help="Target language code to filter by (e.g., 'fr', 'de'). Will filter records and append to output filename.")
    args = parser.parse_args()

    # If keys provided (possibly default), run filter; otherwise run clean only
    if args.keys:
        output_file = filter_jsonl(args.input_path, args.keys, args.min_ft_length, args.max_ft_length, args.ocr_threshold, args.s3_prefix, args.lang)
    else:
        output_file = clean_jsonl_file(args.input_path, args.min_ft_length, args.max_ft_length)

    print(f"Processed file saved to: {output_file}")