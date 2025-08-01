# Adversarial Alignment

Adversarial Alignment is a framework developed for generating textual adversarial examples, for instance jailbreaking LLMs. We introduce Reinforcement Learning from Oracle Feedback (RLOF), an approach similar to RLHF, to adapt a generation policy towards learning how to create adversarial examples. This approach is designed to be agnostic to the adversarial goal, the underlying decision-making process (the oracle part), as well as any and all defenses (for instance guardrails), thus general in its applicability.

## Usage

### Supervised Fine-Tuning

The adversarial example generation process is bootstrapped from known pairs of clean and adversarial examples, as well as a trained model that classifies them. To fine-tune a model for generating adversarial examples, adjust the load_data function and replace the classifier, then run the script:

```bash
python asft.py
```

### RLOF

After ASFT, the model will have learned an initial mapping between clean and adversarial distributions. This does not mean however that the examples that it generates are adversarial - or will continue to be, after adversarial training. To dapt the attacker policy based on the classifier decisions, once more replace the classifier, then run the script:

```bash
python rlof.py
```