import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DistilBertTokenizer, DistilBertForSequenceClassification
)
from trl import GRPOTrainer, GRPOConfig
from asft import load_data
from utils import validate_structure
import gc

"""
Reinforcement Learning from Oracle Feedback (RLOF)
to align an attacker model's adversarial examples
against a web request log classifier.
"""

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

def align_attacker(epochs=1):
    # Load data
    clean_texts, _ = load_data()
    prompts = [p + "~~" for p in clean_texts]
    dataset = Dataset.from_dict({"prompt": prompts})
    # Load classifier (detector)
    detector = DistilBertForSequenceClassification.from_pretrained("models/TFclassifier").to(device).eval()
    tokenizer_det = DistilBertTokenizer.from_pretrained("models/TFclassifier")

    # Load generator (attacker)
    generator = AutoModelForCausalLM.from_pretrained("models/TFadvgen").to(device)
    tokenizer_gen = AutoTokenizer.from_pretrained("models/TFadvgen", padding_side="left")

    def reward_func(completions, **kwargs):
        """
        Compute rewards for a batch of generated texts.
        Args: samples (List[str]): List of generated completions.    
        Returns: List[float]: Rewards for each sample.
        """
        # Tokenize entire batch at once
        inputs = tokenizer_det(
            completions,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            pad_to_multiple_of=8
        ).to(device)

        # Classify in batch
        with torch.no_grad():
            logits = detector(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, 1].cpu().numpy()  # class 1 = bot

        # Reward shaping
        rewards = []
        for log, score in zip(completions, scores):
            reward = 1 - abs(score)
            penalty = validate_structure(log)
            rewards.append(reward - penalty)

        return rewards

    # GRPO configuration
    config = GRPOConfig(
        output_dir="models/TFadapted",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        bf16=True,
        gradient_checkpointing=False,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        temperature=0.3,
        beta=0.1,
        warmup_ratio=0.1,
        weight_decay=0.01, 
        logging_steps=128,
        save_steps=0,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=42,
    )

    # GRPO Trainer
    trainer = GRPOTrainer(
        model=generator,
        processing_class=tokenizer_gen,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_func,
    )

    trainer.train()

    trainer.model.save_pretrained("models/TFadapted")
    tokenizer_gen.save_pretrained("models/TFadapted")

    # Garbage collection
    del generator
    del tokenizer_gen
    del detector
    del tokenizer_det
    torch.cuda.empty_cache()
    gc.collect()

def test_attacker(model):
    # Load data
    clean_texts, _ = load_data()
    prompts = [p + "~~" for p in clean_texts]
    dataset = Dataset.from_dict({"prompt": prompts})
    # Load classifier (detector)
    detector = DistilBertForSequenceClassification.from_pretrained("models/TFclassifier").to(device).eval()
    tokenizer_det = DistilBertTokenizer.from_pretrained("models/TFclassifier")

    # Load adapted generator
    generator = AutoModelForCausalLM.from_pretrained("models/" + model).to(device).eval()
    tokenizer_gen = AutoTokenizer.from_pretrained("models/" + model, padding_side="left")
    
    # Test on multiple prompts
    num_test = 2048
    batch_size = 64
    successes = 0
    penalty = 0
    
    adversarials = []
    for i in range(0, num_test, batch_size):
        batch_end = min(i + batch_size, num_test)
        batch_prompts = [dataset["prompt"][j] for j in range(i, batch_end)]
        
        # Generate completions for batch
        inputs = tokenizer_gen(batch_prompts,
                               return_tensors="pt",
                               padding=True,
                               truncation=True).to(device)
        
        with torch.no_grad():
            output_ids = generator.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer_gen.pad_token_id,
                eos_token_id=tokenizer_gen.eos_token_id,
            )
        
        # Extract generated texts
        generated_texts = []
        penalties = []
        prompt_lens = inputs["input_ids"].shape[-1]
        for j in range(output_ids.shape[0]):
            new_ids = output_ids[j, prompt_lens:]
            generated_text = tokenizer_gen.decode(new_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        # Classify the batch of generated texts
        det_inputs = tokenizer_det(
            generated_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            logits = detector(**det_inputs).logits
            probs = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probs, dim=-1)
        
        successes += (predicted_classes == 0).sum().item()
        for txt in generated_texts:
            pen = validate_structure(txt)
            penalties.append(pen)
            penalty += pen

        # Select generations that are missclassified and structurally valid
        for j, (pred, pen) in enumerate(zip(predicted_classes, penalties)):
            if pred == 0 and pen == 0:
                adversarials.append(generated_texts[j])

    # Save adversarial samples
    file_path = "data/advgens.csv"
    parsed = []
    for sample in adversarials:
        fields = sample.strip().split('|')
        values = [field.split(':', 1)[1].strip() if ':' in field else '' for field in fields]
        while len(values) < 11:
            values.append("")
        values = values[:11]
        parsed.append(values)

    # Create DataFrame with specified column names
    columns = ['host', 'as_number', 'client_address', 'method', 'path', 'request_size', 
               'referer', 'content_type', 'user_agent', 'status_code', 'as_organization']

    df = pd.DataFrame(parsed, columns=columns)
    df['label'] = True

    df.to_csv(file_path,index=False)

    # Print ASR & avg. penalty
    attack_success_rate = (successes / num_test) * 100
    print(f"\nAttack Success Rate: {attack_success_rate:.2f}% ({successes}/{num_test})")
    average_penalty = penalty / num_test
    print(f"Average penalty: {average_penalty:.2f}")

    # Garbage collection
    del generator
    del tokenizer_gen
    del detector
    del tokenizer_det
    torch.cuda.empty_cache()
    gc.collect()

    return attack_success_rate, average_penalty

if __name__ == "__main__":
    align_attacker()
    test_attacker('TFadapted') #TFadapted TFadvgen