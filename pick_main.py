import pandas as pd
import json
from tqdm import tqdm
import augmentation.picking as pk
import importlib
importlib.reload(pk)
import asyncio



concept_root = "./data/concept/"
out_concept_root = "./data/outside_concept/"
response_root = "./data/respondent/"
cache_path = "./data/embeddings_cache/us_food_concepts.pkl"

# take the concepts 
with open(concept_root + 'new_cid_concept_us_food.json', 'r') as f:
    food_concepts = json.load(f)

# all concepts 
all_us_food_concepts = pd.read_excel(out_concept_root + '0407_cleaned_us_food_concepts.xlsx')

# open transformed
with open(response_root + 'transformed_0407_id_normal_interview.json', 'r', encoding='utf-8') as f:
    transformed_respondent = json.load(f)

response_table = pd.read_excel(response_root + "response_table_us_food.xlsx")
response_table.drop(columns=['id'], inplace=True)




concept_pk = pk.ConceptPicker(food_concepts = food_concepts, all_us_food_concepts = all_us_food_concepts, transformed_respondent = transformed_respondent, response_table = response_table, cache_path = cache_path)
id_to_processs = list(response_table['ids'].unique())

split_num = 300

for i in tqdm(range(28, split_num)):
    batch_ids = id_to_processs[i*len(id_to_processs)//split_num : (i+1)*len(id_to_processs)//split_num]
    result = asyncio.run(concept_pk.process_all(respondent_ids = batch_ids, show_progress=False))
    result.to_excel(f"./data/pricking_results/picking_result_{i}.xlsx", index=False)

