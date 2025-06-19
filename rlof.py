import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_logs(generator, tokenizer, input_ids, max_new=512, temperature=0.7):
    """
    Generates token trajectories and accumulates logprobs for a batch of prompts
    Returns:
      generated: tensor of token IDs for each sample
      texts: list of decoded texts
      logprobs: tensor of cumulative logprobs
    """
    # batch_size x prompt_len
    batch_size = input_ids.size(0)
    generated = input_ids.clone()
    logprobs = torch.zeros(batch_size, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        for _ in range(max_new):
            outputs = generator(input_ids=generated)
            # Get logits for the last token.
            logits = outputs.logits[:, -1, :]
            # Apply temperature scaling
            logits = logits / temperature
            # Create a categorical distribution
            dist = torch.distributions.Categorical(logits=logits)
            # Sample next token, shape: (batch,)
            next_tokens = dist.sample()

            # For finished sequences, override with pad
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, pad_id),
                next_tokens)

            # Accumulate logprobs for active sequences only
            logprobs = logprobs + torch.where(
                finished,
                torch.zeros_like(logprobs),
                dist.log_prob(next_tokens))

            # Append the sampled token to each trajectory
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)

            # Mark sequences as finished when EOS token is generated
            finished = finished | (next_tokens == tokenizer.eos_token_id)
            if finished.all():
                break

    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return generated, texts, logprobs

def compute_logprobs(generator, generated, prompt_lengths):
    """
    Recompute the logprob of each sampled trajectory under the
    current model parameters, to form the new/old probability ratio
    that drives the PPO loss.
    Returns summed logprobs *after* the prompt for each sequence
    """
    # Inputs are all tokens except the last, targets are all tokens but the first
    inputs = generated[:, :-1]
    targets = generated[:, 1:]
    outputs = generator(inputs) # shape: (batch, length-1, vocab_size)
    logits = outputs.logits
    # Conditional log‐probability distribution over the next token at each position
    logprobs = F.log_softmax(logits, dim=-1)
    # Logprob the model assigned to the token sampled during generation
    token_logprobs = torch.gather(
        logprobs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Exclude the prompt tokens
    new_logprobs = torch.zeros(generated.size(0), device=device)
    for i, p_len in enumerate(prompt_lengths):
        new_logprobs[i] = token_logprobs[i, p_len - 1 :].sum()
    return new_logprobs

def classify_logs(texts, detector, tokenizer, batch_size=16):
    """Classify generated logs with the detector."""
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            pad_to_multiple_of=8
        ).to(device)
        with torch.no_grad():
            outputs = detector(**inputs)
            batch_scores = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_scores.extend(batch_scores)
    return np.array(all_scores)

def compute_reward(log, score):
    reward = 1 - np.abs(score)
    # Check for valid log structure and values
    # penalty = validate_constraints(log)
    penalty = 0
    # print(penalty)
    return reward - penalty
    
def train_attacker_rl(generator, tokenizer_gen, detector, tokenizer_det,
                      clean_texts, lr, num_epochs=10, samples=16, max_new=512,
                      temp=0.7, ppo_epochs=4, clip_eps=0.2):
    
    optimizer = AdamW(generator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        generator.train()

        # Sample a batch of clean logs to use as prompts
        prompts = random.sample(clean_texts, k=samples)
        prompts = [p + "~~" for p in prompts]
        enc = tokenizer_gen(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False
        ).to(device)
        input_ids = enc.input_ids
        prompt_lengths = enc.attention_mask.sum(dim=1).cpu().tolist()

        # Generate trajectories with the current policy
        generated, texts, old_logprobs = generate_logs(
            generator, tokenizer_gen, input_ids,
            max_new=max_new, temperature=temp
        )

        print(texts)

        # Classify and compute rewards
        scores = classify_logs(texts, detector, tokenizer_det)
        rewards = np.array([compute_reward(log, s) for log, s in zip(texts, scores)])
        print("Rewards:", [f"{r:.2f}" for r in rewards])
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        advantages = torch.tensor(advantages, device=device, dtype=torch.float)

        # Adapt policy
        for ppo_epoch in range(ppo_epochs):
            # Recompute log probabilities
            new_logprobs = compute_logprobs(generator, generated, prompt_lengths)
            # Compute the new to old policy ratio
            ratio = torch.exp(new_logprobs - old_logprobs.detach())
            # PPO-clip objective
            loss_unclipped = ratio * advantages
            loss_clipped = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
            ppo_loss = -torch.mean(torch.min(loss_unclipped, loss_clipped))

            optimizer.zero_grad()
            ppo_loss.backward()
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer.step()
            print(f"PPO Epoch {ppo_epoch+1}/{ppo_epochs} | Loss: {ppo_loss.item():.4f} | Grad Norm: {grad_norm:.4f}")


        print(f"Avg Detector Score: {scores.mean():.3f}")
        print("Example Adv:", texts[0])

if __name__ == "__main__":
    # Load data
    clean = load_data()

    # Load detector
    detector = DistilBertForSequenceClassification.from_pretrained("models/TFclassifier").to(device).eval()
    tokenizer_det = DistilBertTokenizer.from_pretrained("models/TFclassifier")

    # Load adversarial generator
    generator = AutoModelForCausalLM.from_pretrained("models/TFadvgen").to(device)
    tokenizer_gen = AutoTokenizer.from_pretrained("models/TFadvgen")
    tokenizer_gen.padding_side = "left"

    # Train attacker with PPO
    train_attacker_rl(generator,
                      tokenizer_gen,
                      detector,
                      tokenizer_det,
                      clean,
                      lr=1e-5,
                      num_epochs=40,
                      samples=8,
                      max_new=512,
                      temp=0.7,
                      ppo_epochs=4,
                      clip_eps=0.2)

    # Save adapted model
    generator.save_pretrained("models/TFadapted")
    tokenizer_gen.save_pretrained("models/TFadapted")
