# llm_synthetic

Text data augmentation using LLM for concept rephrasing.

## Overview

This project provides tools to generate augmented versions of concept texts through LLM-powered rephrasing. The rephrasing preserves all essential information (features, benefits, claims, specifications) while varying the presentation style.

## Features

The `ConceptRephraser` supports multiple transformation dimensions:

| Dimension | Options | Description |
|-----------|---------|-------------|
| **Tone** | `CLINICAL`, `MARKETING`, `CONVERSATIONAL` | Adjusts the writing style |
| **Point of View** | `SECOND_PERSON`, `THIRD_PERSON` | Changes how the reader is addressed |
| **Content Order** | `PROBLEM_FIRST`, `FEATURE_FIRST`, `BENEFIT_FIRST` | Reorders the narrative structure |
| **Length** | `change_length=True/False` | Expands short texts or condenses long ones |

## Quick Start

### 1. Setup

```python
from augmentation.rephrasing import ConceptRephraser, RephraseConfig, Tone, PointOfView, ContentOrder

# Initialize the rephraser
rephraser = ConceptRephraser(model="gpt-4o", temperature=0.5)
```

### 2. Single Concept Rephrasing

```python
config = RephraseConfig(
    change_length=True,
    tone=Tone.CONVERSATIONAL,
    point_of_view=PointOfView.SECOND_PERSON,
    content_order=ContentOrder.BENEFIT_FIRST
)

result = rephraser.rephrase("Your concept text here...", config)
print(result.rephrased_text)
```

### 3. Batch Processing (Multiple Variations)

See `playground_rephrasing.ipynb` for the full workflow:

1. Load concept data from JSON (format: `{id: {"name": "...", "content": "..."}}`):
   ```python
   with open("./data/concept/cid_concept_us_personalcare.json", encoding="utf-8") as f:
       cid_concept = json.load(f)
   
   for k, v in cid_concept.items():
       cid_concept[k] = v['content']
   ```

2. Generate multiple variations per concept:
   ```python
   def multi_rephrase_concept(concept_text, n_variations=8):
       # Randomly combines different config options
       # Returns list of variations with config metadata
       ...
   ```

3. Process all concepts and save results:
   ```python
   variation_cid_concept = {}
   for k, v in tqdm(cid_concept.items()):
       variations = multi_rephrase_concept(v, n_variations=8)
       variation_cid_concept[k] = {i+1: var for i, var in enumerate(variations)}
   
   # Save to JSON
   with open(f"./data/{date}_variation_{cid_name}", 'w', encoding="utf-8") as f:
       json.dump(variation_cid_concept, f, indent=2, ensure_ascii=False)
   ```

## Output Format

The output JSON contains variations for each concept:

```json
{
  "1": {
    "1": {
      "config": {"tone": "conversational", "point_of_view": null, "content_order": "benefit_first", "change_length": true},
      "text": "Rephrased concept text...",
      "word_count": 95
    },
    "2": { ... }
  }
}
```

## Key Guarantees

The rephraser **preserves**:
- Product/concept names
- Key features and specifications
- Core benefits and claims  
- Ingredients, materials, and technologies
- Original language

The rephraser **does NOT**:
- Invent new features or claims
- Include brand names or prices (configurable)
- Exaggerate or alter factual statements

## Requirements

- Python 3.8+
- OpenAI API key (set `LITE_LLM_KEY_ALL` in `.env`)
- Dependencies: `langchain-openai`, `python-dotenv`, `tqdm`