import datasets
from datasets import load_dataset
import os


paired_dataset_path = "paired_prompts.jsonl"


def format_for_alert_sft(example):
    """Format messages for Alert training"""
    formatted_text = {
        "messages": [
            {"role": "user", "content": f'<query>{example["clean_prompt"]}</query>'},
            {"role": "assistant", "content": f'<jailbreak>{example["adv_prompt"]}</jailbreak>'},
        ]
    }

    return formatted_text


paired_dataset = load_dataset("json", data_files=paired_dataset_path)
formatted_dataset = paired_dataset.map(format_for_alert_sft)
formatted_dataset = formatted_dataset.remove_columns(
    ["clean_prompt", "adv_prompt", 'category', 'clean_id']
)


formatted_dataset['train'].to_json("formatted_alert_sft.jsonl")
