import json
import os
import pandas as pd
from tabulate import tabulate

"""

python comparer_tagger.py tm-de-all-v2.0.topic_model_topic_description_filtered_tagged_gpt.jsonl tm-de-all-v2.0.topic_model_topic_description_filtered_tagged_deepseek.jsonl

"""

def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj["id"]] = obj
    return data

def merge_files(gpt_file, deepseek_file):
    # Load both files into dicts by ID
    gpt_data = load_jsonl(gpt_file)
    deepseek_data = load_jsonl(deepseek_file)

    # Ensure they have the same IDs
    if set(gpt_data.keys()) != set(deepseek_data.keys()):
        raise ValueError("The two files do not contain the same IDs")

    disagreements = []
    merged = []

    for _id in gpt_data.keys():
        gpt_tag = gpt_data[_id].get("tag", "unknown")
        deepseek_tag = deepseek_data[_id].get("tag", "unknown")

        if gpt_tag == deepseek_tag:
            final_tag = gpt_tag
        else:
            final_tag = "unknown"  # mark disagreement as unknown
            disagreements.append((_id, gpt_tag, deepseek_tag))

        merged_obj = deepseek_data[_id].copy()
        merged_obj["tag"] = final_tag
        merged.append(merged_obj)

    # Save merged file
    output_path = os.path.splitext(gpt_file)[0].replace("_gpt", "") + "_final.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for obj in merged:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Print summary table
    if disagreements:
        df = pd.DataFrame(disagreements, columns=["ID", "GPT Tag", "DeepSeek Tag"])
        print(f"\nTotal disagreements: {len(disagreements)}\n")
        print(tabulate(df, headers="keys", tablefmt="grid"))
    else:
        print("\nNo disagreements found!")

    print(f"\nMerged file saved as: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge GPT and DeepSeek tagged JSONL files")
    parser.add_argument("gpt_file", help="Path to GPT-tagged JSONL file")
    parser.add_argument("deepseek_file", help="Path to DeepSeek-tagged JSONL file")
    args = parser.parse_args()

    merge_files(args.gpt_file, args.deepseek_file)
