#!/usr/bin/env python3
import argparse, json, sys
from collections import defaultdict

"""

python collect_non_ads.py tm-de-all-v2.0.topic_model_topic_description_filtered_tagged_final.jsonl tm-fr-all-v2.0.topic_model_topic_description_filtered_tagged_final.jsonl tm-lb-all-v2.1.topic_model_topic_description_filtered_tagged_final.jsonl -o non_ads.json

"""

def extract_lang(topic_id: str) -> str:
    """
    Language is the second hyphen-separated token in the id.
    Example: 'tm-de-all-v2.0_tp00_de' -> 'de'
    """
    parts = topic_id.split("-")
    return parts[1] if len(parts) >= 2 else ""

def main():
    parser = argparse.ArgumentParser(
        description="Collect topic IDs tagged as 'non-ad' from JSONL files, grouped by language."
    )
    parser.add_argument("inputs", nargs="+", help="Input .jsonl file(s)")
    parser.add_argument("-o", "--out", default="non_ads_by_language.json",
                        help="Output JSON file (default: non_ads_by_language.json)")
    args = parser.parse_args()

    non_ads_by_lang = defaultdict(set)  # use set to avoid duplicates
    total_lines = 0

    for path in args.inputs:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # skip malformed lines
                        continue

                    tag = obj.get("tag")
                    topic_id = obj.get("id")
                    if tag == "non-ad" and topic_id:
                        lang = extract_lang(topic_id)
                        if lang:
                            non_ads_by_lang[lang].add(topic_id)
        except FileNotFoundError:
            print(f"Warning: file not found: {path}", file=sys.stderr)
        except OSError as e:
            print(f"Warning: couldn't read {path}: {e}", file=sys.stderr)

    # convert sets to sorted lists for JSON output
    output = {lang: sorted(list(ids)) for lang, ids in non_ads_by_lang.items()}

    with open(args.out, "w", encoding="utf-8") as out_f:
        json.dump(output, out_f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output)} languages to {args.out} "
          f"(from ~{total_lines} lines across {len(args.inputs)} file(s)).")

if __name__ == "__main__":
    main()
