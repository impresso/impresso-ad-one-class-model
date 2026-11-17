#!/usr/bin/env python3
"""Compare local ad labels with an OpenAI model and split into agree/disagree JSONL files.

Expected input row shape:
{"id": "AATA-1846-04-18-a-i0035", "ad_classification": "non-ad", "ft": "text ..."}

Behavior:
- If OpenAI agrees with `ad_classification`, write the original row unchanged to the agree file.
- If OpenAI disagrees, write the full row to the disagree file and add
  `openai_classification` next to `ad_classification`.

Example:
python data_preparation_second_finetuning/openai_compare_ad_classification.py \
  --input-file ads-nonads_inspection_sample.jsonl
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import requests
from dotenv import load_dotenv
from smart_open import open as smart_open


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COOKBOOK_LIB_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "cookbook", "lib")
if COOKBOOK_LIB_DIR not in sys.path:
    sys.path.insert(0, COOKBOOK_LIB_DIR)

try:
    from common import get_transport_params
except ImportError:
    def get_transport_params(path: str) -> Dict[str, Any]:
        return {}


LOG = logging.getLogger(__name__)

load_dotenv()

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OUTPUT_KEY = "openai_classification"
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}

CLASSIFICATION_INSTRUCTIONS = (
    "You classify historical newspaper texts into one of two labels only: "
    "'ad' or 'non-ad'. "
    "Return 'ad' if the text is any form of advertisement, commercial notice, "
    "classified ad, service offer, product offer, recruitment ad, auction ad, "
    "or promotional notice. "
    "Return 'non-ad' otherwise. "
    "Use only the provided text. Do not explain your answer."
)

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["ad", "non-ad"],
        }
    },
    "required": ["classification"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare existing ad labels with an OpenAI model and split into agree/disagree JSONL files."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Input JSONL file path (local or S3).",
    )
    parser.add_argument(
        "--agree-output",
        help="Output JSONL for rows where OpenAI agrees. Defaults to <input>_agree.jsonl",
    )
    parser.add_argument(
        "--disagree-output",
        help="Output JSONL for rows where OpenAI disagrees. Defaults to <input>_disagree.jsonl",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--label-key",
        default="ad_classification",
        help="Input key containing the existing label (default: ad_classification)",
    )
    parser.add_argument(
        "--text-key",
        default="ft",
        help="Input key containing the text to classify (default: ft)",
    )
    parser.add_argument(
        "--output-key",
        default=DEFAULT_OUTPUT_KEY,
        help=f"Key added to disagree rows for the OpenAI label (default: {DEFAULT_OUTPUT_KEY})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional cap on how many valid rows to process.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log progress every N processed rows (default: 25)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="HTTP timeout per request in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per OpenAI request (default: 5)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the whole run on a row that still fails after retries.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Start fresh by truncating existing agree/disagree output files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def normalize_label(value: object) -> Optional[str]:
    if value is None:
        return None

    normalized = str(value).strip().lower().replace("_", "-")
    if normalized == "ad":
        return "ad"
    if normalized in {"non-ad", "nonad"}:
        return "non-ad"
    return None


def derive_output_paths(
    input_path: str,
    agree_output: Optional[str],
    disagree_output: Optional[str],
) -> Tuple[str, str]:
    if agree_output and disagree_output:
        return agree_output, disagree_output

    base, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".jsonl"

    return (
        agree_output or f"{base}_agree{ext}",
        disagree_output or f"{base}_disagree{ext}",
    )


def ensure_local_parent(path: str) -> None:
    if path.startswith("s3://"):
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    transport_params = get_transport_params(path)
    with smart_open(path, "r", encoding="utf-8", transport_params=transport_params) as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_no, json.loads(stripped)
            except json.JSONDecodeError as exc:
                LOG.warning("Skipping invalid JSON at %s:%s: %s", path, line_no, exc)


def write_jsonl_row(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def load_processed_ids(*paths: str, id_key: str = "id") -> Set[str]:
    processed_ids: Set[str] = set()

    for path in paths:
        transport_params = get_transport_params(path)
        try:
            with smart_open(path, "r", encoding="utf-8", transport_params=transport_params) as handle:
                for line_no, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        LOG.warning("Skipping invalid JSON in existing output %s:%s: %s", path, line_no, exc)
                        continue

                    row_id = row.get(id_key)
                    if not isinstance(row_id, str) or not row_id.strip():
                        LOG.warning("Skipping existing output row without valid %s at %s:%s", id_key, path, line_no)
                        continue

                    processed_ids.add(row_id)
        except FileNotFoundError:
            continue
        except OSError as exc:
            LOG.warning("Could not read existing output %s: %s", path, exc)

    return processed_ids


def add_field_after(
    row: Dict[str, Any],
    after_key: str,
    new_key: str,
    new_value: Any,
) -> Dict[str, Any]:
    updated: Dict[str, Any] = {}
    inserted = False

    for key, value in row.items():
        updated[key] = value
        if key == after_key:
            updated[new_key] = new_value
            inserted = True

    if not inserted:
        updated[new_key] = new_value

    return updated


def extract_output_text(response_json: Dict[str, Any]) -> str:
    chunks = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content_item in item.get("content", []):
            if content_item.get("type") == "output_text":
                text = content_item.get("text")
                if isinstance(text, str):
                    chunks.append(text)

    if not chunks:
        raise ValueError("OpenAI response did not contain output_text")

    return "".join(chunks).strip()


def classify_text(
    session: requests.Session,
    api_key: str,
    base_url: str,
    model: str,
    text: str,
    request_timeout: int,
    max_retries: int,
) -> str:
    payload = {
        "model": model,
        "store": False,
        "instructions": CLASSIFICATION_INSTRUCTIONS,
        "input": f"Classify this text:\n\n{text}",
        "max_output_tokens": 32,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ad_classification_result",
                "strict": True,
                "schema": CLASSIFICATION_SCHEMA,
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = base_url.rstrip("/") + "/responses"

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=request_timeout,
            )

            if response.status_code in RETRYABLE_STATUS_CODES:
                request_id = response.headers.get("x-request-id", "unknown")
                raise RuntimeError(
                    f"retryable OpenAI error {response.status_code} "
                    f"(request_id={request_id}): {response.text[:500]}"
                )

            response.raise_for_status()
            response_json = response.json()

            if response_json.get("error"):
                raise RuntimeError(f"OpenAI returned error: {response_json['error']}")

            output_text = extract_output_text(response_json)
            parsed = json.loads(output_text)
            classification = normalize_label(parsed.get("classification"))
            if classification is None:
                raise ValueError(f"Unexpected classification payload: {parsed}")
            return classification

        except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            sleep_seconds = min(30, 2 ** (attempt - 1))
            LOG.warning(
                "OpenAI request failed on attempt %s/%s: %s. Retrying in %ss.",
                attempt,
                max_retries,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"OpenAI classification failed after {max_retries} attempts: {last_error}")


def main() -> None:
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    if args.max_rows is not None and args.max_rows <= 0:
        raise SystemExit("--max-rows must be greater than 0")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be greater than 0")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be greater than 0")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be greater than 0")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)-15s %(filename)s:%(lineno)d %(levelname)s: %(message)s",
    )

    agree_output, disagree_output = derive_output_paths(
        input_path=args.input_file,
        agree_output=args.agree_output,
        disagree_output=args.disagree_output,
    )
    ensure_local_parent(agree_output)
    ensure_local_parent(disagree_output)

    processed_ids: Set[str] = set()
    if not args.overwrite_output:
        processed_ids = load_processed_ids(agree_output, disagree_output)
        if processed_ids:
            LOG.info("Loaded %s already-processed ids from existing outputs.", len(processed_ids))

    stats = {
        "lines_seen": 0,
        "valid_rows_processed": 0,
        "agree": 0,
        "disagree": 0,
        "classification_failures": 0,
        "skipped_already_processed": 0,
        "skipped_missing_text": 0,
        "skipped_invalid_label": 0,
    }

    with requests.Session() as session:
        agree_transport = get_transport_params(agree_output)
        disagree_transport = get_transport_params(disagree_output)
        output_mode = "w" if args.overwrite_output else "a"

        with smart_open(
            agree_output,
            output_mode,
            encoding="utf-8",
            transport_params=agree_transport,
        ) as agree_handle, smart_open(
            disagree_output,
            output_mode,
            encoding="utf-8",
            transport_params=disagree_transport,
        ) as disagree_handle:
            for line_no, row in iter_jsonl(args.input_file):
                stats["lines_seen"] += 1

                row_id = row.get("id")
                if isinstance(row_id, str) and row_id in processed_ids:
                    stats["skipped_already_processed"] += 1
                    continue

                local_label = normalize_label(row.get(args.label_key))
                if local_label is None:
                    stats["skipped_invalid_label"] += 1
                    LOG.warning(
                        "Skipping line %s: invalid %s=%r",
                        line_no,
                        args.label_key,
                        row.get(args.label_key),
                    )
                    continue

                ft = row.get(args.text_key)
                if not isinstance(ft, str) or not ft.strip():
                    stats["skipped_missing_text"] += 1
                    LOG.warning(
                        "Skipping line %s: missing/empty %s",
                        line_no,
                        args.text_key,
                    )
                    continue

                try:
                    openai_label = classify_text(
                        session=session,
                        api_key=api_key,
                        base_url=args.base_url,
                        model=args.model,
                        text=ft.strip(),
                        request_timeout=args.request_timeout,
                        max_retries=args.max_retries,
                    )
                except RuntimeError as exc:
                    stats["classification_failures"] += 1
                    LOG.error(
                        "Classification failed for line %s id=%r: %s",
                        line_no,
                        row.get("id"),
                        exc,
                    )
                    if args.stop_on_error:
                        raise
                    continue

                stats["valid_rows_processed"] += 1
                if openai_label == local_label:
                    write_jsonl_row(agree_handle, row)
                    stats["agree"] += 1
                else:
                    disagree_row = add_field_after(
                        row=row,
                        after_key=args.label_key,
                        new_key=args.output_key,
                        new_value=openai_label,
                    )
                    write_jsonl_row(disagree_handle, disagree_row)
                    stats["disagree"] += 1
                if isinstance(row_id, str) and row_id:
                    processed_ids.add(row_id)

                if stats["valid_rows_processed"] % args.progress_every == 0:
                    LOG.info(
                        "Processed %s new valid rows: agree=%s disagree=%s classification_failures=%s skipped_already_processed=%s",
                        stats["valid_rows_processed"],
                        stats["agree"],
                        stats["disagree"],
                        stats["classification_failures"],
                        stats["skipped_already_processed"],
                    )

                if args.max_rows is not None and stats["valid_rows_processed"] >= args.max_rows:
                    LOG.info("Reached --max-rows=%s, stopping early.", args.max_rows)
                    break

    LOG.info("Finished.")
    LOG.info("Input file: %s", args.input_file)
    LOG.info("Model: %s", args.model)
    LOG.info("Agree output: %s", agree_output)
    LOG.info("Disagree output: %s", disagree_output)
    LOG.info("Lines seen: %s", stats["lines_seen"])
    LOG.info("Valid rows processed: %s", stats["valid_rows_processed"])
    LOG.info("Agree: %s", stats["agree"])
    LOG.info("Disagree: %s", stats["disagree"])
    LOG.info("Classification failures: %s", stats["classification_failures"])
    LOG.info("Skipped already processed: %s", stats["skipped_already_processed"])
    LOG.info("Skipped invalid labels: %s", stats["skipped_invalid_label"])
    LOG.info("Skipped missing/empty texts: %s", stats["skipped_missing_text"])


if __name__ == "__main__":
    main()
