#!/usr/bin/env python3
"""
Random JSONL sampler - extracts a random subset of lines from a JSONL file.

Usage:
    python jsonl_random_sampler.py --input_file input.jsonl --output_file output.jsonl --sample_size 1000 --seed 42
"""
import argparse
import json
import random
import sys
from typing import List


def sample_jsonl(input_file: str, output_file: str, sample_size: int) -> None:
    """
    Randomly sample lines from a JSONL file and save to a new JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file  
        sample_size: Number of lines to sample
    """
    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total lines in input: {total_lines}")
    
    if sample_size > total_lines:
        print(f"Warning: Requested {sample_size} samples but only {total_lines} lines available")
        sample_size = total_lines
    
    # Randomly sample lines
    sampled_lines = random.sample(lines, sample_size)
    
    # Write sampled lines to output
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in sampled_lines:
            f.write(line)
    
    print(f"Successfully sampled {len(sampled_lines)} lines to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample lines from a JSONL file",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input_file", required=True, help="Input JSONL file path")
    parser.add_argument("--output_file", required=True, help="Output JSONL file path")
    parser.add_argument("--sample_size", type=int, required=True, help="Number of lines to sample")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible results")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    try:
        sample_jsonl(args.input_file, args.output_file, args.sample_size)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()