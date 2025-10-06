import json
import sys

def rename_ci_id_to_id(input_file, output_file):
    """Rename 'ci_id' field to 'id' in JSON lines file"""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if line:
                data = json.loads(line)
                if 'ci_id' in data:
                    data['id'] = data.pop('ci_id')
                outfile.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rename_ci_id.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    rename_ci_id_to_id(input_file, output_file)
    print(f"Renamed 'ci_id' to 'id' in {input_file} -> {output_file}")