import argparse
import sys
from adaptation.rlof import GRPOPipeline, ScriptArguments, GPUConfig, LoRAArgs


def detect_gpus():
    """Find GPUs with >16GB free memory. Returns (victim_gpu, attacker_gpu)."""
    from utils import get_free_gpus

    gpus = get_free_gpus(min_free_gb=16.0)
    if not gpus:
        print("No GPUs with enough free memory found.")
        sys.exit(1)

    for gpu_id, free_gb in gpus:
        print(f"  GPU {gpu_id}: {free_gb:.1f} GB free")

    if len(gpus) >= 2:
        victim, attacker = gpus[0][0], gpus[1][0]
        print(f"Dual GPU mode: victim=GPU {victim}, attacker+judge=GPU {attacker}")
    else:
        victim = attacker = gpus[0][0]
        print(f"Single GPU mode: all models on GPU {victim}")

    return victim, attacker


def main():
    parser = argparse.ArgumentParser(description="GRPO training for adversarial alignment")
    parser.add_argument("--eval-mode", choices=["rule_based", "llm_judge", "hybrid"], default="hybrid")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--victim-model", default="meta-llama/Llama-3.1-8B-Instruct", help="Victim model name from registry")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--verbose", action="store_true", help="Print jailbreak examples during training")
    parser.add_argument("--verbose-every", type=int, default=10, help="Print a verbose sample every N batches (default: 1)")
    args = parser.parse_args()

    victim_gpu, attacker_gpu = detect_gpus()

    print(f"\nStarting GRPO training (eval={args.eval_mode}, judge={args.judge_model})")

    script_args = ScriptArguments(reward_funcs=["accuracy"])
    gpu_args = GPUConfig(
        generator_gpu=attacker_gpu,
        victim_gpu=victim_gpu,
        evaluator_gpu=attacker_gpu,
        evaluator_mode=args.eval_mode,
        judge_model=args.judge_model,
        victim_model_name=args.victim_model,
        verbose=args.verbose,
        verbose_every=args.verbose_every,
    )
    lora_args = LoRAArgs(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    pipeline = GRPOPipeline(script_args, gpu_args, lora_args)
    pipeline.setup()
    pipeline.train()


if __name__ == "__main__":
    main()
