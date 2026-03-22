import json

def load_jsonl(file_path):
    eval_data = []
    # Open the JSONL file
    with open(file_path, 'r', encoding='utf-8') as file:
        # Iterate over each line in the file
        for line in file:
            # Parse the JSON object
            json_obj = json.loads(line)
            # Do something with the JSON object
            eval_data.append(json_obj)
    return eval_data


def save_list_to_jsonl(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            # json.dumps converts the dictionary to a JSON string
            f.write(json.dumps(item) + '\n')
