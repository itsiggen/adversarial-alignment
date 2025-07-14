import json
import nltk
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(
    description="Analyze prompts using a chosen model, with an optional processing limit.",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("input_file", type=str, help="Path to the input .jsonl file.")
parser.add_argument(
    "--model-type",
    type=str,
    choices=['marco', 'deberta'],
    default='marco',
    help="The type of model to use:\n"
         "  'marco': Fast relevance ranker (default)\n"
         "  'deberta': Slower, high-accuracy NLI model"
)
parser.add_argument(
    "--threshold",
    type=float,
    help="Custom threshold for a match"
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Stop processing after this many items (lines) from the input file."
)
args = parser.parse_args()

if args.model_type == 'marco':
    MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    THRESHOLD = args.threshold if args.threshold is not None else 0
    print(f"Using MARCO model with relevance threshold: {THRESHOLD}")
else: # 'deberta'
    MODEL_NAME = 'cross-encoder/nli-deberta-v3-large'
    THRESHOLD = args.threshold if args.threshold is not None else .9
    print(f"Using DeBERTa NLI model with entailment threshold: {THRESHOLD}")

INPUT_FILE = args.input_file
base_name = os.path.basename(INPUT_FILE).replace('.jsonl', '')
model_tag = f"{args.model_type}_limit{args.limit}" if args.limit else args.model_type

MATCHING_OUTPUT_FILE = f"{base_name}_{model_tag}_1_successful_matches.jsonl"
SECONDARY_NON_MATCHING_OUTPUT_FILE = f"{base_name}_{model_tag}_2_secondary_non_matches.jsonl"
PROMPTS_WITH_NO_MATCHES_OUTPUT_FILE = f"{base_name}_{model_tag}_3_prompts_with_no_matches.jsonl"

print(f"Loading the '{MODEL_NAME}' model...")
model = CrossEncoder(MODEL_NAME)
print("Model loaded.")

if not os.path.exists(INPUT_FILE):
    print(f"Error: Input file '{INPUT_FILE}' not found.")
    exit()

matching_results, secondary_non_matching_results, prompts_with_no_matches = [], [], []
item_counter = 0

print(f"\nProcessing items from '{INPUT_FILE}'...")
with open(INPUT_FILE, 'r') as f_in:
    for line in f_in:
        if args.limit is not None and item_counter >= args.limit:
            print(f"\nReached processing limit of {args.limit} items. Halting.")
            break
        item_counter += 1
        
        item = json.loads(line)
        clean_id, clean_prompt, adv_prompt = item.get('clean_id'), item.get('clean_prompt'), item.get('adv_prompt')

        print(f"  - Analyzing item #{item_counter} (id: {clean_id})")
        if not all([clean_prompt, adv_prompt]): continue

        sentences = nltk.sent_tokenize(adv_prompt)
        if not sentences: continue
            
        sentence_pairs = [[sentence, clean_prompt] for sentence in sentences]
        
        if args.model_type == 'marco':
            scores = model.predict(sentence_pairs)
        else: # deberta
            scores = model.predict(sentence_pairs, apply_softmax=True)

        current_item_matches, current_item_non_matches = [], []

        for i, sentence in enumerate(sentences):
            score_to_check = 0.0
            result_obj = {"sentence": sentence}

            if args.model_type == 'marco':
                score_to_check = scores[i]
                result_obj["relevance_score"] = float(score_to_check)
            else: # deberta
                score_to_check = scores[i][1] # Entailment score
                result_obj["contradiction_score"] = float(scores[i][0])
                result_obj["entailment_score"] = float(score_to_check)
                result_obj["neutral_score"] = float(scores[i][2])
            
            if score_to_check > THRESHOLD:
                current_item_matches.append(result_obj)
            else:
                current_item_non_matches.append(result_obj)

        # Categorization Logic
        if current_item_matches:
            matching_results.append({"clean_id": clean_id, "matches_found": current_item_matches})
            if current_item_non_matches:
                secondary_non_matching_results.append({"clean_id": clean_id, "secondary_non_matches": current_item_non_matches})
        else:
            prompts_with_no_matches.append({"clean_id": clean_id, "non_matches": current_item_non_matches})

def write_results_to_file(filepath, data_list):
    with open(filepath, 'w') as f_out:
        for item in data_list: f_out.write(json.dumps(item) + '\n')

write_results_to_file(MATCHING_OUTPUT_FILE, matching_results)
write_results_to_file(SECONDARY_NON_MATCHING_OUTPUT_FILE, secondary_non_matching_results)
write_results_to_file(PROMPTS_WITH_NO_MATCHES_OUTPUT_FILE, prompts_with_no_matches)

print("\nAnalysis complete.")
print(f"  - Successful matches saved to:         '{MATCHING_OUTPUT_FILE}'")
print(f"  - Fluff from successful prompts saved to: '{SECONDARY_NON_MATCHING_OUTPUT_FILE}'")
print(f"  - Prompts with no matches saved to:     '{PROMPTS_WITH_NO_MATCHES_OUTPUT_FILE}'")