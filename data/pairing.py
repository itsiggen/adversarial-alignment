import json
from tqdm import tqdm
import nltk as nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Load datasets
with open("alert.jsonl", "r", encoding="utf-8") as f:
    clean_data = [json.loads(line) for line in f]

with open("alert_adversarial.jsonl", "r", encoding="utf-8") as f:
    adv_data = [json.loads(line) for line in f]

# Extract and tokenize
def extract_prompt(text):
    return text.split("### Instruction:\n")[1].split("\n### Response:")[0].strip()

for entry in clean_data:
    entry["tokens"] = word_tokenize(extract_prompt(entry["prompt"]).lower())

for entry in adv_data:
    entry["tokens"] = word_tokenize(extract_prompt(entry["prompt"]).lower())

# Index clean prompts by category
clean_by_category = {}
for idx, entry in enumerate(clean_data):
    cat = entry["category"]
    clean_by_category.setdefault(cat, []).append((idx, entry))

# Similarity funcs
def token_overlap(tokens1, tokens2):
    set1, set2 = set(tokens1), set(tokens2)
    return len(set1 & set2)

def jaccard(tokens1, tokens2):
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)

# Match adversarial prompts to clean ones
paired_dataset = []

for i, adv in tqdm(enumerate(adv_data), total=len(adv_data)):
    adv_tokens = adv["tokens"]
    category = adv["category"]
    candidates = clean_by_category.get(category, [])

    sims = []
    for clean_idx, clean in candidates:
        sim = jaccard(adv_tokens, clean["tokens"])
        sims.append({
            "prompt": extract_prompt(clean["prompt"]),
            "id": clean_idx,
            "similarity": sim
        })

    # Select highest similarity
    matches = sorted(sims, key=lambda x: x["similarity"])
    # print(adv, matches[-2:])

    paired_dataset.append({
        "category": category,
        "clean_prompt": matches[-1]["prompt"],
        "clean_id": matches[-1]["id"],
        "adv_prompt": extract_prompt(adv["prompt"])
    })

# Save to JSONL
with open("paired_prompts.jsonl", "w", encoding="utf-8") as f:
    for item in paired_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
