import json
import sys
from pathlib import Path


"""
python extract_top_words.py tm-de-all-v2.0.topic_model_topic_description.jsonl

"""

def extract_top_words(input_file, top_n=50):
    input_path = Path(input_file)
    output_file = input_path.with_name(input_path.stem + "_filtered.jsonl")

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            record = json.loads(line)
            
            # Sort by prob just in case input isn’t sorted
            top_words = [
                wp["word"]
                for wp in sorted(record.get("word_probs", []),
                                 key=lambda x: x["prob"],
                                 reverse=True)[:top_n]
            ]
            
            new_record = {
                "id": record["id"],
                "top_20_words": top_words
            }
            outfile.write(json.dumps(new_record, ensure_ascii=False) + "\n")

    print(f"Filtered file saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_top_words.py <input_file.jsonl>")
        sys.exit(1)

    extract_top_words(sys.argv[1])
