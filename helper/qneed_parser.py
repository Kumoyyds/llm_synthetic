"""
Parser for extracting qneed2/qneed3 information from survey responses.
"""
import re
import json
from typing import Dict, List, Optional, Any


def extract_qneed_entries(response_text: str) -> List[Dict[str, str]]:
    """
    Extract qneed entries (category and comment) from the qneed23 section.
    
    Args:
        response_text: The full survey response text
        
    Returns:
        List of dicts with 'cate' and 'comment' keys
    """
    qneed_entries = []
    
    # Find the qneed23 section
    qneed23_match = re.search(r'----qneed23----(.+?)(?=----\w+----|\Z)', response_text, re.DOTALL)
    
    if not qneed23_match:
        return qneed_entries
    
    qneed23_content = qneed23_match.group(1)
    
    # Pattern 1: "regarding of [Category],What aspects are you frustrated by..."
    # Pattern 2: "And what more would you like [Category] products to do better in..."
    # Pattern 3: Just extract any q: and a: pairs in the section
    
    # Find all q: a: pairs
    qa_pattern = r'q:(.+?)\na:(.+?)(?=\nq:|\n\n\n|\Z)'
    qa_matches = re.findall(qa_pattern, qneed23_content, re.DOTALL)
    
    for question, answer in qa_matches:
        question = question.strip()
        answer = answer.strip()
        
        # Skip empty answers
        if not answer:
            continue
            
        # Extract category from different question patterns
        category = None
        
        # Pattern: "regarding of [Category],What aspects..."
        match1 = re.search(r'regarding of ([^,]+),', question, re.IGNORECASE)
        if match1:
            category = match1.group(1).strip()
        
        # Pattern: "And what more would you like [Category] products to do better"
        if not category:
            match2 = re.search(r'would you like (.+?)(?:\s+products)? to do better', question, re.IGNORECASE)
            if match2:
                category = match2.group(1).strip()
        
        # Pattern: "Thinking of [Category], how satisfied..."
        if not category:
            match3 = re.search(r'Thinking of ([^,]+),', question, re.IGNORECASE)
            if match3:
                category = match3.group(1).strip()
        
        if category:
            qneed_entries.append({
                'cate': category,
                'comment': answer
            })
    
    return qneed_entries


def transform_respondent_data(id_response: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Transform the respondent data to include qneed2/qneed3 labels.
    
    Args:
        id_response: Dict mapping respondent ID to their survey response text
        
    Returns:
        Dict with structure:
        {
            'id': {
                'response': 'original response text',
                'qneed2': {'cate': '...', 'comment': '...'},
                'qneed3': {'cate': '...', 'comment': '...'},
                ...
            }
        }
    """
    transformed = {}
    
    for respondent_id, response_text in id_response.items():
        # Extract qneed entries
        qneed_entries = extract_qneed_entries(response_text)
        
        # Build the transformed entry
        entry = {
            'response': response_text
        }
        
        # Add qneed entries with labels qneed2, qneed3, etc.
        for i, qneed in enumerate(qneed_entries, start=2):
            entry[f'qneed{i}'] = qneed
        
        transformed[respondent_id] = entry
    
    return transformed


def load_and_transform(input_path: str, output_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load respondent data from JSON file, transform it, and optionally save to a new file.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Optional path to save the transformed data
        
    Returns:
        Transformed data dict
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        id_response = json.load(f)
    
    transformed = transform_respondent_data(id_response)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transformed, f, indent=2, ensure_ascii=False)
    
    return transformed


# Example usage
if __name__ == "__main__":
    import os
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_path = os.path.join(project_dir, "data", "respondent", "0331_id_normal_interview.json")
    output_path = os.path.join(project_dir, "data", "respondent", "0331_id_with_qneed.json")
    
    result = load_and_transform(input_path, output_path)
    
    # Print sample output
    for resp_id, data in list(result.items())[:2]:
        print(f"\nID: {resp_id}")
        print(f"Keys: {list(data.keys())}")
        for key in data:
            if key.startswith('qneed'):
                print(f"  {key}: {data[key]}")
