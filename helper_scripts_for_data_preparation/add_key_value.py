import json
import sys
import argparse

'''
python add_key_value.py input.jsonl output.jsonl "type" "ad"
'''

def add_key_value_to_jsonl(input_file, output_file, key, value):
    """Add a key-value pair to each line in a JSONL file"""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    data[key] = value
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")
                    continue

def main():
    parser = argparse.ArgumentParser(
        description='Add a key-value pair to each line in a JSONL file',
        epilog='''
Examples:
  %(prog)s input.jsonl output.jsonl "type" "ad"
  %(prog)s data.jsonl labeled_data.jsonl "score" "85" --type int
  %(prog)s texts.jsonl processed.jsonl "is_valid" "true" --type bool
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_file', help='Input JSONL file')
    parser.add_argument('output_file', help='Output JSONL file')
    parser.add_argument('key', help='Key to add')
    parser.add_argument('value', help='Value to add')
    parser.add_argument('--type', choices=['str', 'int', 'float', 'bool'], default='str',
                       help='Type of the value (default: str)')
    
    args = parser.parse_args()
    
    # Convert value to the specified type
    if args.type == 'int':
        try:
            value = int(args.value)
        except ValueError:
            print(f"Error: Cannot convert '{args.value}' to int")
            sys.exit(1)
    elif args.type == 'float':
        try:
            value = float(args.value)
        except ValueError:
            print(f"Error: Cannot convert '{args.value}' to float")
            sys.exit(1)
    elif args.type == 'bool':
        value = args.value.lower() in ('true', '1', 'yes', 'on')
    else:
        value = args.value
    
    try:
        add_key_value_to_jsonl(args.input_file, args.output_file, args.key, value)
        print(f"Successfully added key '{args.key}' with value '{value}' to all lines")
        print(f"Output written to: {args.output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
