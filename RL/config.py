from dataclasses import dataclass, field
from typing import List, Optional

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class GRPOScriptArguments:
    """
    Script arguments for GRPO training, specifically related to reward functions.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'cosine'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Minimum reward for cosine scaling for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.1,
        metadata={"help": "Maximum reward for cosine scaling for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.8,
        metadata={"help": "Minimum reward for cosine scaling for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for cosine scaling for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for cosine scaling"},
    )

@dataclass
class ModelConfig:
    """
    Configuration for the model.
    """
    model_name_or_path: str = field(
        default=MODEL_NAME, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_revision: Optional[str] = field(
        default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "Override the default `torch_dtype` and load the model under this dtype."}
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Trust remote code when loading model and tokenizer."}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "Attention implementation to use. 'flash_attention_2' or None"}
    )