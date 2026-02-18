import logging
import os
import torch
from dataclasses import dataclass, field
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer, ProgressCallback, set_seed
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

from .rewards import get_reward_functions, shutdown_evaluator, _stats

logger = logging.getLogger(__name__)

# Defaults — base model must match what SFT was trained on
BASE_MODEL = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = "data/Qwen-GRPO-LoRA-training"
DATASET_PATH = "finetune/data/combined_pairs.jsonl"
SFT_ADAPTER_PATH = "finetune/data/Qwen-LoRAd"


@dataclass
class ScriptArguments:
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy"],
        metadata={"help": "Reward functions to use"},
    )
    cosine_min_value: float = field(default=0.5)
    cosine_max_value: float = field(default=1.0)
    cosine_max_len: int = field(default=1000)


@dataclass
class GPUConfig:
    generator_gpu: int = field(default=0, metadata={"help": "GPU for attacker/generator model"})
    victim_gpu: int = field(default=1, metadata={"help": "GPU for victim model (largest)"})
    evaluator_gpu: int = field(default=0, metadata={"help": "GPU for judge model"})
    evaluator_mode: str = field(default="hybrid", metadata={"help": "rule_based, llm_judge, or hybrid"})
    judge_model: str = field(default="Qwen/Qwen2.5-3B-Instruct", metadata={"help": "Judge model ID"})
    victim_model_name: str = field(default=None, metadata={"help": "Victim model name from registry"})
    eval_timeout: float = field(default=120.0)
    use_length_scaling: bool = field(default=True)
    verbose: bool = field(default=False, metadata={"help": "Print one sample per reward batch"})
    verbose_every: int = field(default=1, metadata={"help": "Print a verbose sample every N batches"})


@dataclass
class LoRAArgs:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


class PipelineProgressCallback(ProgressCallback):
    """Extends default progress bar to show rolling jailbreak success rate."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and _stats.total > 0:
            logs["jb%"] = round(_stats.rate * 100, 1)
        return super().on_log(args, state, control, logs=logs, **kwargs)


class GRPOPipeline:
    """End-to-end GRPO training pipeline for adversarial jailbreak generation."""

    def __init__(self, script_args: ScriptArguments, gpu_args: GPUConfig, lora_args: LoRAArgs):
        self.script_args = script_args
        self.gpu_args = gpu_args
        self.lora_args = lora_args

        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None

        # Filled during _setup_gpus
        self.gen_device = None
        self.victim_device = None
        self.eval_device = None

    def _setup_gpus(self):
        """Make needed GPUs visible and build physical -> logical mapping."""
        gpu = self.gpu_args
        unique_gpus = list(dict.fromkeys([gpu.generator_gpu, gpu.victim_gpu, gpu.evaluator_gpu]))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in unique_gpus)
        gpu_map = {phys: logical for logical, phys in enumerate(unique_gpus)}

        self.gen_device = gpu_map[gpu.generator_gpu]
        self.victim_device = gpu_map[gpu.victim_gpu]
        self.eval_device = gpu_map[gpu.evaluator_gpu]
        torch.cuda.set_device(self.gen_device)
        logger.info(
            f"GPUs: generator=cuda:{self.gen_device}, "
            f"victim=cuda:{self.victim_device}, judge=cuda:{self.eval_device}"
        )

    def _load_model(self):
        """Load base model, merge SFT adapter, then apply fresh LoRA."""
        # Tokenizer (from SFT adapter if available, else base model)
        tok_path = SFT_ADAPTER_PATH if os.path.exists(SFT_ADAPTER_PATH) else BASE_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Base model
        logger.info(f"Loading base model {BASE_MODEL}")
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map={"": self.gen_device},
        )

        # Merge SFT adapter if available
        if os.path.exists(SFT_ADAPTER_PATH):
            logger.info(f"Merging SFT adapter from {SFT_ADAPTER_PATH}")
            self.model = PeftModel.from_pretrained(self.model, SFT_ADAPTER_PATH)
            self.model = self.model.merge_and_unload()
        else:
            logger.warning("No SFT adapter found — training from base model")

        # Fresh LoRA for GRPO training
        la = self.lora_args
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=la.lora_r, lora_alpha=la.lora_alpha, lora_dropout=la.lora_dropout,
            target_modules=la.lora_target_modules, bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def _load_dataset(self):
        """Load and format the prompt dataset for RL."""
        logger.info(f"Loading dataset from {DATASET_PATH}")
        self.dataset = load_dataset("json", data_files=DATASET_PATH, cache_dir=OUTPUT_DIR)

        def format_for_rl(example):
            return {
                "prompt": f"Original: {example['input']}\nJailbreak:",
                "original_query": example["input"],
            }

        self.dataset = self.dataset.map(format_for_rl)
        self.dataset = self.dataset.remove_columns(["input", "output"])

    def _build_reward_functions(self):
        """Create reward functions with pipeline kwargs baked in."""
        gpu = self.gpu_args
        sa = self.script_args
        reward_kwargs = {
            "evaluator_mode": gpu.evaluator_mode,
            "evaluator_gpu_id": self.eval_device,
            "victim_gpu_id": self.victim_device,
            "judge_model": gpu.judge_model,
            "victim_model_name": gpu.victim_model_name,
            "eval_timeout": gpu.eval_timeout,
            "use_length_scaling": gpu.use_length_scaling,
            "cosine_min_value": sa.cosine_min_value,
            "cosine_max_value": sa.cosine_max_value,
            "cosine_max_len": sa.cosine_max_len,
            "verbose": gpu.verbose,
            "verbose_every": gpu.verbose_every,
        }

        raw = get_reward_functions(sa)

        def wrap(func):
            def wrapped(*args, **kwargs):
                return func(*args, **{**reward_kwargs, **kwargs})
            return wrapped

        return [wrap(f) for f in raw]

    def _build_trainer(self, reward_functions):
        """Configure GRPOTrainer with the model, dataset, and rewards."""
        grpo_config = GRPOConfig(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=3,
            dataloader_num_workers=2,
            seed=42,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            num_generations=4,
        )

        # Train only on generator GPU
        grpo_config._n_gpu = 1
        logger.info(f"Forced n_gpu=1 (visible GPUs: {torch.cuda.device_count()})")

        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_functions,
            args=grpo_config,
            train_dataset=self.dataset["train"],
        )

        # Replace default progress bar with one showing jailbreak success rate
        self.trainer.remove_callback(ProgressCallback)
        self.trainer.add_callback(PipelineProgressCallback())

    def setup(self):
        # Run all setup steps
        set_seed(60)
        self._setup_gpus()
        self._load_model()
        self._load_dataset()
        reward_functions = self._build_reward_functions()
        self._build_trainer(reward_functions)

    def train(self):
        # Run GRPO training and save the resulting LoRA adapters.
        logger.info("Starting GRPO training")
        try:
            self.trainer.train()
            self.tokenizer.save_pretrained(OUTPUT_DIR)
            self.model.save_pretrained(OUTPUT_DIR)
            logger.info(f"LoRA adapters saved to {OUTPUT_DIR}")
        finally:
            shutdown_evaluator()
