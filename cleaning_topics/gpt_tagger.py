import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import os


"""
python gpt_tagger.py tm-de-all-v2.0.topic_model_topic_description_filtered.jsonl

"""


# Load variables from .env file
load_dotenv()

from openai import OpenAI

# Initialize with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def classify_topic(words):
    """Ask GPT whether the words represent advertisement, non-ad, or unknown."""
    prompt = (
        "You are given top words that represent a topic from mallet LDA topic modeling on historical articles from 19th and 20th centuries. "
        "Decide if the topic is likely:\n"
        "- Advertisement (output 'ad')\n"
        "- Non-advertisement (output 'non-ad')\n"
        "- Difficult to say (output 'unknown')\n\n"
        f"Words: {', '.join(words)}\n\n"
        "Answer with only one of: ad, non-ad, unknown."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # you can change to another GPT model if you prefer
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    tag = response.choices[0].message.content.strip().lower()
    if tag not in {"ad", "non-ad", "unknown"}:
        tag = "unknown"  # fallback
    return tag


def process_file(input_path):
    """Process the JSONL file and save with _tagged_gpt suffix."""
    output_path = os.path.splitext(input_path)[0] + "_tagged_gpt.jsonl"

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            data = json.loads(line)
            words = data.get("top_20_words", [])
            tag = classify_topic(words)
            data["tag"] = tag
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    print(f"Saved tagged file: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tag JSONL topics as ad / non-ad / unknown using GPT.")
    parser.add_argument("input_file", help="Path to input JSONL file")
    args = parser.parse_args()
    process_file(args.input_file)
