import os
import torch
import datasets
import argparse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


# Configuration
MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = "data/Qwen-LoRAd"
DATASET_PATH = "data/combined_pairs.jsonl"


def format_example(example, tokenizer):
    """Format with prompt masking - only train on the jailbreak output"""
    input_text = example["input"]
    output_text = example["output"]
    
    prompt = f"Original: {input_text}\nJailbreak: "
    full_text = prompt + output_text + tokenizer.eos_token
    
    # Tokenize separately to get lengths
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    
    # Create labels: -100 for prompt (masked), actual tokens for output
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids)
    }


def train_model():
    """Train the model with LoRA"""
    LORA_CONFIG = LoraConfig(
        r=32,  # Rank -- 64 for better performance
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Jailbreak Fine-Tuning...")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Apply LoRA
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Load dataset
    dataset = datasets.load_dataset("json", data_files=DATASET_PATH)
    print(f"Dataset size: {len(dataset['train'])} examples")

    formatted_dataset = dataset.map(
        lambda x: format_example(x, tokenizer), 
        remove_columns=["input", "output"]
    )

    # Data collator for padding variable-length sequences
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # Training configuration
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=1024,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        eval_strategy="no",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=42,
        report_to="none",
        remove_unused_columns=False,  # Keep our custom columns
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset["train"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Train
    train_result = trainer.train()

    # Save model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nModel saved to {OUTPUT_DIR}")
    print(f"Training loss: {train_result.training_loss:.4f}")

    return model, tokenizer


def generate_jailbreaks(samples=5, model=None, tokenizer=None):
    """Generate example jailbreaks using the trained model"""

    # Load model if not provided
    if model is None or tokenizer is None:
        print(f"Loading trained model from {OUTPUT_DIR}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            OUTPUT_DIR,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
        model.eval()
        print("Model loaded successfully!\n")

    # Load dataset
    original_dataset = datasets.load_dataset("json", data_files=DATASET_PATH)["train"]

    # Prepare model for inference
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()

    for i in range(samples):
        original_question = original_dataset[i]["input"]
        prompt = f"Original: {original_question}\nJailbreak:"
        
        print(f"Example {i+1}:")
        print(f"Original: {original_question}")
        print(f"\nGenerated Jailbreak:")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        jailbreak = generated_text.split("Jailbreak:")[-1].strip()
        print(jailbreak)
        print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test jailbreak model")
    parser.add_argument("--test", action="store_true", help="Load trained model and generate jailbreaks")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to generate")
    args = parser.parse_args()

    if args.test:
        # Test
        generate_jailbreaks(args.samples)
    else:
        # Train
        model, tokenizer = train_model()
        # Generate examples after training
        generate_jailbreaks(args.samples, model, tokenizer)