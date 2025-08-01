import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import pipeline

"""
Adversarial supervised fine-tuning (ASFT) to generate
adversarial examples against a web request log classifier.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and preprocess the data
def load_data():
    # Load clean data
    to_drop = ['response_size', 'cookie', 'network', 'request_body', 'pattern', 'label']
    clean = pd.read_json("data/processed.json")
    clean = clean.drop(to_drop, axis=1)

    num_cols = ['client_address', 'request_size']
    cat_cols = [col for col in clean.columns if col not in num_cols]

    for col in cat_cols:
        clean[col] = clean[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
   
    # Convert to text
    def concat_text(row):
        tagged_text = []
        for col in row.index:
            value = row[col]
            tagged_text.append(f"{col}: {value}")
        return "|".join(tagged_text) + "|"

    clean_texts = clean.apply(concat_text, axis=1)

    # Load adversarial data
    adv = pd.read_csv('data/advs.csv')
    og_idx = adv["og_index"]
    adv = adv.drop(['label','og_index'], axis=1)
    adv_texts = adv.apply(concat_text, axis=1).tolist()

    # Align clean and adversarial by index
    mapped_clean = og_idx.map(clean_texts)
    mask = mapped_clean.notna()
    clean_texts = mapped_clean[mask].tolist()

    return clean_texts, adv_texts

def tokenize_and_mask(dataset, tokenizer, max_length=1024):
    sep = "~~"
    eos = tokenizer.eos_token

    def fn(examples):
        # List to hold full input strings
        inputs = []
        # List to hold tokenized lengths of the clean+sep portion
        prefix_lens = []
        for clean, adv in zip(examples["clean"], examples["adv"]):
            # Tokenize only the prefix (clean + sep) to get its token length
            prefix_ids = tokenizer(clean + sep, add_special_tokens=False)["input_ids"]
            prefix_lens.append(len(prefix_ids))
            # full text = clean + separator + adv + <eos>
            inputs.append(clean + sep + adv + eos)

        # Tokenize the full inputs, with padding and truncation
        enc = tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            add_special_tokens=False,
        )

        # Mask out prefix tokens to remove from loss calculation
        labels = []
        for ids, p_len in zip(enc["input_ids"], prefix_lens):
            lab = ids.copy()
            lab[:p_len] = [-100] * p_len
            labels.append(lab)
        enc["labels"] = labels
        return enc

    return dataset.map(fn, batched=True, remove_columns=["clean", "adv"])

# Fine-tune for generating adversarial examples, given clean ones as prompts
def fine_tune(clean_texts, adversarial_texts):
    ds = Dataset.from_dict({"clean": clean_texts, "adv": adversarial_texts})

    # Load model and tokenizer
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False)

    # Add pad token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    # Tokenize & mask labels
    tokenized = tokenize_and_mask(ds, tokenizer)

    training_args = TrainingArguments(
        output_dir="models/TFadvgen",
        overwrite_output_dir=True,
        num_train_epochs=8,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_strategy="no",
        prediction_loss_only=True,
        fp16=True,
        learning_rate=1e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("models/TFadvgen")
    tokenizer.save_pretrained("models/TFadvgen")

def generate_adversarial(clean_samples, num_return=1):
    model = AutoModelForCausalLM.from_pretrained("models/TFadvgen").to(device)
    tokenizer = AutoTokenizer.from_pretrained("models/TFadvgen")
    tokenizer.padding_side = "left"

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
    )

    for clean in clean_samples:
        prompt = clean + "~~"
        outs = generator(
            prompt,
            max_length=1512,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=num_return,
            eos_token_id=tokenizer.eos_token_id,
        )
        print(f"\n=== CLEAN ===\n{clean}\n--- ADVERSARIAL GEN ---")
        for i, o in enumerate(outs, 1):
            adv = o["generated_text"][len(prompt):]
            print(f"{i}) {adv}")

if __name__ == "__main__":
    cleans, advs = load_data()
    fine_tune(cleans, advs)
    generate_adversarial(cleans[:15], num_return=2)
