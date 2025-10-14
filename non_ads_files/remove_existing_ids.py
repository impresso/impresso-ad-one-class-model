import json

def remove_matching_ids(file_103, file_4000):
    # Load the ids from the first JSONL file
    ids_to_remove = set()
    with open(file_103, 'r') as f103:
        for line in f103:
            data = json.loads(line)
            ids_to_remove.add(data.get('id'))

    # Load the second JSONL file, filter out lines with matching IDs
    remaining_lines = []
    with open(file_4000, 'r') as f4000:
        for line in f4000:
            data = json.loads(line)
            if data.get('id') not in ids_to_remove:
                remaining_lines.append(line)

    # Overwrite the second JSONL file with the filtered lines
    with open(file_4000, 'w') as f4000:
        f4000.writelines(remaining_lines)

# Specify file paths
file_103 = 'non_ads_103_for_hyperparameters.jsonl'
file_4000 = 'non_ads_4000_finetuning_.jsonl'

# Call the function
remove_matching_ids(file_103, file_4000)
