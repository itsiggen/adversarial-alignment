# Adversarial Alignment

A general framework for automating the red-teaming of LLMs and agentic workflows. Our environment is composed of an attacker, a victim, and a judge model. The attacker is initially fine-tuned on pairs of forbidden prompts and their jailbroken versions. Subsequently, attacker generations are fed as prompts to the victim model. These responses together with the inital prompt are then evaluated by the judge model, to identify if they are successful jailbreaks. The last part provides a reward signal on which the attacker adapts its generation policy with GRPO. The framework is agnostic to the victiom model and any defenses or guardrails, and in this way generally applicable.

## Pipeline

```
Dataset (input/output pairs)
    |
    +-- Step 1: SFT --- Attacker learns jailbreak patterns from examples
    |
    +-- Step 2: GRPO -- Attacker optimizes against live victim + judge
    |                     |
    |                     +-- Attacker generates jailbreak from prompt
    |                     +-- Victim model responds to the jailbreak
    |                     +-- Judge evaluates whether the victim complied
    |                     +-- Reward signal updates attacker policy
    |
    +-- Step 3: Generate & Evaluate
```

## Setup

```bash
pip install -r requirements.txt
```

Note: vllm is only needed if you use the `llm_judge` or `hybrid` evaluation modes during GRPO training. If you only use `rule_based` evaluation, you can skip it.

## Usage

### 1. SFT Pre-training

Train the attacker to learn basic jailbreak patterns via supervised fine-tuning:

```bash
python finetune/sft_conditional.py
```

To test the trained model:

```bash
python finetune/sft_conditional.py --test --samples 5
```

### 2. GRPO Training

Run the full Attacker -> Victim -> Judge policy adaptation loop:

```bash
# Auto-detect GPUs and run with defaults (hybrid evaluation)
python train.py

# Specify evaluation mode and judge model
python train.py --eval-mode hybrid --judge-model Qwen/Qwen2.5-3B-Instruct

# Choose a specific victim model
python train.py --victim-model mistral-7b
```

### 3. Generation

Generate example jailbreaks with the trained attacker:

```bash
# Generate 5 example jailbreaks (default)
python generate.py
```

### 4. Evaluation

Test the trained attacker against a victim model and measure success rate:

```bash
# Evaluate against default victim (mistral-7b)
python generate.py evaluate --num 10

# Evaluate against a specific victim model from the registry
python generate.py evaluate --victim-model qwen2.5-3b --num 10

# Evaluate against any HuggingFace model by ID
python generate.py evaluate --victim-id Qwen/Qwen2.5-7B-Instruct --num 10

# Use a different GPU for the victim model
python generate.py evaluate --victim-model mistral-7b --victim-device cuda:1
```

## Model Selection

### Attacker Model

Default: `Qwen/Qwen2.5-1.5B-Instruct`. Configured at the top of `adaptation/rlof.py`.

### Victim Models

Set via `--victim-model` on `train.py` or `generate.py evaluate`. Available from the registry:

| Name | Model | Size | Auth |
|------|-------|------|------|
| mistral-7b | Mistral-7B-Instruct-v0.2 | 14.5 GB | No |
| qwen2.5-7b | Qwen2.5-7B-Instruct | 15 GB | No |
| qwen2.5-3b | Qwen2.5-3B-Instruct | 6.5 GB | No |
| phi3-mini | Phi-3-mini-4k-instruct | 8 GB | No |
| llama3-8b | Meta-Llama-3-8B-Instruct | 16 GB | Yes |
| llama2-7b | Llama-2-7B-Chat | 14 GB | Yes |
| gemma-7b | Gemma-7B-it | 17 GB | Yes |

Default: mistral-7b. Add more models in `victim/model_registry.py`.

### Judge Models

Set via `--judge-model` (any HuggingFace model ID). Default: `Qwen/Qwen2.5-3B-Instruct`.

The judge evaluates victim responses to determine whether a jailbreak was successful.

### Evaluation Modes

| Mode | Description |
|------|-------------|
| hybrid (default) | Rule-based for clear cases, LLM judge for ambiguous ones |
| rule_based | Fast string-matching only (uses jailbreakeval library) |
| llm_judge | LLM-based evaluation for all samples (slower, most accurate) |
