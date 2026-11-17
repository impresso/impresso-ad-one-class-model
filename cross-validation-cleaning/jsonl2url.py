import json
import os

# --- Configuration ---
INPUT_FILENAME = 'predictions/non_ads_predictions_dissagree.jsonl'
OUTPUT_FILENAME = 'predictions_non_ads_dissagree_urls.txt'
BASE_URL = 'https://impresso-project.ch/app/article/'
# ---------------------

def generate_urls_from_jsonl(input_file, output_file, base_url):
    """
    Reads a JSONL file, extracts the 'id' from each line, and writes a
    constructed URL to a new text file.
    """
    print(f"Starting conversion from {input_file} to {output_file}...")
    url_count = 0

    try:
        # Open the input file for reading and the output file for writing
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:

            # Iterate over each line in the input JSONL file
            for line_number, line in enumerate(infile, 1):
                try:
                    # Parse the JSON string from the line
                    data = json.loads(line.strip())

                    # Extract the 'id' field
                    document_id = data.get('id')

                    if document_id:
                        # Construct the full URL
                        url = f"{base_url}{document_id}\n"
                        
                        # Write the URL to the output file
                        outfile.write(url)
                        url_count += 1
                    else:
                        print(f"Warning: 'id' key missing or empty in line {line_number}. Skipping.")

                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON format found in line {line_number}. Skipping line.")
                except Exception as e:
                    print(f"An unexpected error occurred processing line {line_number}: {e}")

        print(f"\nSuccessfully generated {url_count} URLs.")
        print(f"Output saved to: {os.path.abspath(output_file)}")

    except FileNotFoundError:
        print(f"\nError: Input file not found at '{input_file}'.")
        print("Please ensure 'input.jsonl' exists in the same directory as the script.")
    except Exception as e:
        print(f"\nAn error occurred during file operation: {e}")


if __name__ == "__main__":
    # Note: For this script to run, you need to create an 'input.jsonl' file
    # with your data in the same directory.
    generate_urls_from_jsonl(INPUT_FILENAME, OUTPUT_FILENAME, BASE_URL)
