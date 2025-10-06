#!/usr/bin/env python3
"""
Simple script to extract article IDs from JSONL and create impresso URLs.

Usage:
    python extract_urls.py input.jsonl output.txt
"""

import json
import sys
import argparse

def extract_urls(input_jsonl, output_txt):
    """Extract article IDs and create URLs."""
    urls = []
    
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                article_id = data.get('id')
                
                if article_id:
                    url = f"https://impresso-project.ch/app/article/{article_id}"
                    urls.append(url)
                else:
                    print(f"Warning: No 'id' field found in line {line_num}")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in line {line_num}: {e}")
                continue
    
    # Write URLs to output file
    with open(output_txt, 'w', encoding='utf-8') as f:
        for url in urls:
            f.write(url + '\n')
    
    print(f"Extracted {len(urls)} URLs from {input_jsonl}")
    print(f"URLs saved to {output_txt}")

def main():
    parser = argparse.ArgumentParser(description='Extract article URLs from JSONL file')
    parser.add_argument('input_jsonl', help='Input JSONL file')
    parser.add_argument('output_txt', help='Output text file with URLs')
    
    args = parser.parse_args()
    
    try:
        extract_urls(args.input_jsonl, args.output_txt)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_jsonl}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
