import json
import random

input_file = 'paired_prompts.jsonl'
output_file = 'antipairs.jsonl'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        entry = json.loads(line)
        words = entry['clean_prompt'].split()

        # Randomly choose a subset of words, ensuring at least one remains
        num_to_keep = random.randint(1, len(words))
        # To preserve word order, sort by original index
        selected_indices = sorted(random.sample(range(len(words)), num_to_keep))
        shortened = ' '.join([words[i] for i in selected_indices])

        entry['adv_prompt'] = shortened
        outfile.write(json.dumps(entry) + '\n')