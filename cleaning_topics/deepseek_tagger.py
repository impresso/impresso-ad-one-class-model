import json
import os
from dotenv import load_dotenv
from openai import OpenAI

"""
python deepseek_tagger.py tm-de-all-v2.0.topic_model_topic_description_filtered.jsonl

"""

# Load from .env
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("Please set the DEEPSEEK_API_KEY environment variable (e.g. in .env)")

# Initialize DeepSeek client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def classify_topic_deepseek(words):
    """Ask DeepSeek whether the words represent advertisement, non-ad, or unknown."""
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
        model="deepseek-chat",  # or you could use "deepseek-reasoner" if you prefer more reasoning
        messages=[{"role": "system", "content": "You are a helpful classifier."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )
    # Extract the assistant’s response
    content = response.choices[0].message.content.strip().lower()
    if content in {"ad", "non-ad", "unknown"}:
        return content
    else:
        # fallback in case response is unexpected
        return "unknown"

def process_file(input_path):
    """Process the JSONL file using DeepSeek and save with _tagged_deepseek suffix."""
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_tagged_deepseek.jsonl"

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:100]}...")
                continue
            
            words = data.get("top_20_words", [])
            tag = classify_topic_deepseek(words)
            data["tag"] = tag
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Saved tagged file: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tag JSONL topics as ad / non-ad / unknown using DeepSeek.")
    parser.add_argument("input_file", help="Path to input JSONL file")
    args = parser.parse_args()
    process_file(args.input_file)
