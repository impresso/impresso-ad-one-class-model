#!/usr/bin/env python3
import sys
import random
from pathlib import Path

N_REMOVE = 150  # how many lines to drop

def main(path: str):
    file_path = Path(path)

    # Read all lines
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)

    if not lines:
        print("File is empty, nothing to remove.")
        return

    # If file has fewer than N_REMOVE lines, remove as many as possible
    n_remove = min(N_REMOVE, len(lines))

    # Choose indices to remove
    indices_to_remove = set(random.sample(range(len(lines)), n_remove))

    # Keep all lines whose index is not selected
    kept_lines = [
        line for i, line in enumerate(lines) if i not in indices_to_remove
    ]

    # Overwrite the original file
    file_path.write_text("".join(kept_lines), encoding="utf-8")
    print(f"Removed {n_remove} lines from {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path/to/file.jsonl")
        sys.exit(1)

    main(sys.argv[1])
