# Adversarial Alignment

Automated red-teaming framework for LLMs. An attacker model learns to generate jailbreaks that elicit harmful responses from a victim model, guided by a judge that scores each attempt.

## Pipeline

```
1. SFT      Attacker learns jailbreak patterns from example pairs
2. GRPO     Attacker optimizes against live victim + judge
               Attacker generates jailbreak from forbidden prompt
               Victim responds to the jailbreak
               Judge evaluates whether the victim complied
               Topic relevance gate filters off-topic false positives
               Reward signal updates attacker policy
3. Generate  Produce and evaluate jailbreaks with trained model
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. SFT Pre-training

```bash
python finetune/sft_conditional.py
python finetune/sft_conditional.py --test --samples 5  # test
```

### 2. GRPO Training

```bash
python train.py --verbose                           # print samples during training
python train.py --victim-model llama3.1-8b          # specific victim
```

### 3. Generate

```bash
python generate.py                                  # generate example jailbreaks
```

## Project Structure

```
train.py                  GRPO training entry point
generate.py               Generation and evaluation
adaptation/
  rlof.py                 GRPO training pipeline
  rewards.py              Reward functions and topic relevance gate
judge/
  evaluator.py            Rule-based, LLM, and hybrid evaluators
victim/
  victim_model.py         Victim model wrapper
  model_registry.py       Supported victim models
finetune/
  sft_conditional.py      SFT pre-training
```