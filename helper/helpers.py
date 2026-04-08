import json
import re

def is_english_simple(data:str) -> bool:
    """
    Simple version: checks if 'Language:en' exists in the data
    """
    # Find the language code between '[Language:' and '=>'
    if not isinstance(data, str):
        return False
    start = data.find('[Language:')
    if start == -1:
        return False
    
    end = data.find('=>', start)
    if end == -1:
        return False
    
    language_code = data[start + 10:end].strip().lower()
    
    # Check if it starts with 'en'
    return language_code.startswith('en')

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


def extract_concept_info(concept_string: str) -> str:
    """
    Extract key information from a concept string.
    Removes the language tag prefix (e.g., '[Language:en-us=>') and trailing bracket.
    
    Args:
        concept_string: String in format '[Language:xx-xx=>content]'
    
    Returns:
        The extracted content without the language tag wrapper.
    """
    # Pattern matches [Language:xx-xx=> at start and ] at end
    match = re.match(r'\[Language:[a-z]{2}-[a-z]{2}=>(.*)\]$', concept_string, re.DOTALL)
    if match:
        return match.group(1)
    return concept_string  # Return original if pattern doesn't match

def clean_category_name(category_string: str) -> str:
    """
    Clean category names by removing numeric prefixes/labels.
    
    Args:
        category_string: Category string with numeric prefix (e.g., "4.14 Salty snacks & appetizers")
    
    Returns:
        Cleaned category name without the numeric prefix (e.g., "Salty snacks & appetizers")
    
    Examples:
        "4.14 Salty snacks & appetizers" → "Salty snacks & appetizers"
        "4.14.3 Dry savory snacks & chips" → "Dry savory snacks & chips"
    """
    # Pattern matches digits with dots (e.g., 4.14 or 4.14.3) followed by whitespace at the start
    match = re.match(r'^\d+(\.\d+)*\s+(.*)$', category_string.strip())
    if match:
        return match.group(2)
    return category_string.strip()  # Return original if no numeric prefix found
