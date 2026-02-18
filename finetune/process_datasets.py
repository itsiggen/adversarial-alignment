import json
import re
from pathlib import Path


def extract_query_and_jailbreak(messages):
    """
    Extract the original query and jailbreak from the messages format.
    Returns (original_query, jailbreak_version) or (None, None) if parsing fails.
    """
    try:
        # Find the user message with <query>...</query>
        user_msg = next(m for m in messages if m["role"] == "user")
        assistant_msg = next(m for m in messages if m["role"] == "assistant")
        
        # Extract content between <query> tags
        query_match = re.search(r'<query>(.*?)</query>', user_msg["content"], re.DOTALL)
        if not query_match:
            return None, None
        original_query = query_match.group(1).strip()
        
        # Extract content between <jailbreak> tags
        jailbreak_match = re.search(r'<jailbreak>(.*?)</jailbreak>', assistant_msg["content"], re.DOTALL)
        if not jailbreak_match:
            # Some entries have "None" as the jailbreak
            if assistant_msg["content"].strip() == "<jailbreak>None</jailbreak>":
                return None, None
            return None, None
        
        jailbreak_version = jailbreak_match.group(1).strip()
        
        # Skip entries where jailbreak is just "None"
        if jailbreak_version.lower() == "none":
            return None, None
            
        return original_query, jailbreak_version
        
    except (StopIteration, KeyError):
        return None, None


def process_dataset(input_path, output_path):
    """
    Process a jailbreak dataset and create simple input-output pairs.
    Format: {"input": "original question", "output": "jailbreak version"}
    """
    print(f"Processing {input_path}...")
    
    pairs = []
    skipped = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                original, jailbreak = extract_query_and_jailbreak(data["messages"])
                
                if original and jailbreak:
                    pairs.append({
                        "input": original,
                        "output": jailbreak
                    })
                else:
                    skipped += 1
                    
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num}")
                skipped += 1
                continue
    
    # Save as JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"✓ Processed {len(pairs)} pairs (skipped {skipped})")
    print(f"✓ Saved to {output_path}\n")
    
    return len(pairs)


def main():
    data_dir = Path("data")
    
    # Process both datasets
    datasets = [
        ("semantic_prompt_dataset_sft.json", "semantic_pairs.jsonl"),
        ("token_prompt_dataset_sft.json", "token_pairs.jsonl"),
    ]
    
    total_pairs = 0
    
    for input_file, output_file in datasets:
        input_path = data_dir / input_file
        output_path = data_dir / output_file
        
        if input_path.exists():
            count = process_dataset(input_path, output_path)
            total_pairs += count
        else:
            print(f"Warning: {input_path} not found, skipping...")
    
    # Create a combined dataset
    combined_path = data_dir / "combined_pairs.jsonl"
    print(f"Creating combined dataset: {combined_path}")
    
    with open(combined_path, 'w', encoding='utf-8') as outf:
        for _, output_file in datasets:
            output_path = data_dir / output_file
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as inf:
                    for line in inf:
                        outf.write(line)
    
    print(f"Total pairs: {total_pairs}")

if __name__ == "__main__":
    main()
