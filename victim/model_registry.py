from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class VictimModelConfig:
    """Configuration for a victim model."""
    name: str
    model_id: str
    requires_auth: bool = False

# Registry of available victim models
VICTIM_MODELS: Dict[str, VictimModelConfig] = {
    "llama3.1-8b": VictimModelConfig(
        name="Llama-3.1-8B-Instruct",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        requires_auth=True
    ),
    "llama3-8b": VictimModelConfig(
        name="Llama-3-8B-Instruct",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        requires_auth=True  # Requires HuggingFace token
    ),
    "llama3.2-3b": VictimModelConfig(
        name="Llama-3.2-3B-Instruct",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        requires_auth=True
    ),
    "mistral-7b": VictimModelConfig(
        name="Mistral-7B-Instruct-v0.2",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        requires_auth=False
    ),
    "qwen2.5-7b": VictimModelConfig(
        name="Qwen2.5-7B-Instruct",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        requires_auth=False
    ),
    "gemma-7b": VictimModelConfig(
        name="Gemma-7B-it",
        model_id="google/gemma-7b-it",
        requires_auth=True  # Requires accepting license
    ),
    "qwen2.5-3b": VictimModelConfig(
        name="Qwen2.5-3B-Instruct",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        requires_auth=False
    ),
}


def get_victim_model(name: str) -> Optional[VictimModelConfig]:
    """Get victim model configuration by name."""
    return VICTIM_MODELS.get(name)