"""Unified training CLI for EfficientQAT.

Orchestrates Block-AP and E2E-QP phases with wandb logging.
Both phases log to a single wandb run, with metrics prefixed by phase
(block_ap/... and e2e_qp/...) so they show as separate sections.

Usage:
    python train.py \
        --model ./models/Llama-2-7b-hf \
        --net Llama-2 \
        --wbits 4 --group_size 128 \
        --scheme baseline \
        --phases block_ap,e2e_qp \
        --dataset redpajama
"""
import argparse
import os
import subprocess
import sys

from dotenv import load_dotenv


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="EfficientQAT unified training pipeline")

    # Model
    parser.add_argument("--model", type=str, required=True, help="Path to full-precision model")
    parser.add_argument("--net", type=str, default=None, help="Model family name (e.g. Llama-2)")

    # Quantization
    parser.add_argument("--wbits", type=int, default=4, help="Weight quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="Quantization group size")

    # Training
    parser.add_argument("--phases", type=str, default="block_ap,e2e_qp",
                        help="Comma-separated phases to run: block_ap, e2e_qp, or both")
    parser.add_argument("--dataset", type=str, default="redpajama", help="Training dataset")
    parser.add_argument("--dataset_format", type=str, default="pt", help="Dataset format for E2E-QP")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per block (Block-AP)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for Block-AP")
    parser.add_argument("--e2e_batch_size", type=int, default=1, help="Batch size for E2E-QP")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Grad accum for E2E-QP")
    parser.add_argument("--training_seqlen", type=int, default=2048, help="Sequence length for Block-AP")
    parser.add_argument("--pt_context_len", type=int, default=4096, help="Context length for E2E-QP")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Epochs for E2E-QP")

    # Learning rates
    parser.add_argument("--quant_lr", type=float, default=1e-4, help="LR for quantization params (Block-AP)")
    parser.add_argument("--weight_lr", type=float, default=1e-5, help="LR for weights (Block-AP)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="LR for E2E-QP")

    # Experiment
    parser.add_argument("--scheme", type=str, default="baseline", help="Experiment scheme name")
    parser.add_argument("--wandb_project", type=str, default="qat", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name (auto-generated if not set)")

    # Evaluation
    parser.add_argument("--eval_ppl", action="store_true", default=True, help="Evaluate perplexity")
    parser.add_argument("--eval_tasks", type=str, default="piqa,arc_easy,arc_challenge,hellaswag,winogrande",
                        help="lm-eval tasks")

    # Paths
    parser.add_argument("--output_dir", type=str, default="./output", help="Base output directory")
    parser.add_argument("--seed", type=int, default=2, help="Random seed")

    args = parser.parse_args()

    if args.net is None:
        args.net = os.path.basename(args.model.rstrip('/'))

    # Validate wandb API key
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("ERROR: WANDB_API_KEY not found.")
        print("Set it in .env file or export WANDB_API_KEY=<your_key>")
        sys.exit(1)

    import wandb
    wandb.login(key=api_key)

    # Create a single wandb run for the entire experiment
    quant_config = f"w{args.wbits}g{args.group_size}"
    run_name = args.wandb_run_name or f"{args.scheme}-{args.net}-{quant_config}"

    config = {
        "scheme": args.scheme,
        "model_family": args.net,
        "wbits": args.wbits,
        "group_size": args.group_size,
        "quant_config": quant_config,
        "dataset": args.dataset,
        "phases": args.phases,
        "quant_lr": args.quant_lr,
        "weight_lr": args.weight_lr,
        "learning_rate": args.learning_rate,
        "block_ap_epochs": args.epochs,
        "block_ap_batch_size": args.batch_size,
        "block_ap_seqlen": args.training_seqlen,
        "e2e_batch_size": args.e2e_batch_size,
        "e2e_context_len": args.pt_context_len,
        "e2e_epochs": args.num_train_epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
    }
    tags = [args.scheme, args.net, quant_config, args.dataset]

    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=config,
        tags=tags,
    )
    run_id = run.id
    print(f"wandb run created: {run_name} (id={run_id})")
    print(f"Project: {args.wandb_project}")
    run.finish()

    phases = [p.strip() for p in args.phases.split(",")]

    # Paths for intermediate artifacts
    block_ap_output = os.path.join(args.output_dir, "block_ap_log", f"{args.net}-{quant_config}")
    block_ap_model = os.path.join(args.output_dir, "block_ap_models", f"{args.net}-{quant_config}")
    e2e_output = os.path.join(args.output_dir, "e2e_qp_models", f"{args.net}-{quant_config}-{args.dataset}")

    if "block_ap" in phases:
        print(f"\n{'='*60}")
        print(f"Phase 1: Block-AP | {args.net} | {quant_config}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, "main_block_ap.py",
            "--model", args.model,
            "--output_dir", block_ap_output,
            "--net", args.net,
            "--wbits", str(args.wbits),
            "--group_size", str(args.group_size),
            "--quant_lr", str(args.quant_lr),
            "--weight_lr", str(args.weight_lr),
            "--real_quant",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--training_seqlen", str(args.training_seqlen),
            "--calib_dataset", args.dataset,
            "--seed", str(args.seed),
            "--save_quant_dir", block_ap_model,
            "--wandb_project", args.wandb_project,
            "--wandb_run_id", run_id,
            "--scheme", args.scheme,
        ]
        if args.eval_ppl:
            cmd.append("--eval_ppl")
        if args.eval_tasks:
            cmd.extend(["--eval_tasks", args.eval_tasks])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Block-AP failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    if "e2e_qp" in phases:
        print(f"\n{'='*60}")
        print(f"Phase 2: E2E-QP | {args.net} | {quant_config}")
        print(f"{'='*60}\n")

        lr = args.learning_rate
        if args.wbits == 2 and lr == 1e-5:
            lr = 2e-5

        cmd = [
            sys.executable, "main_e2e_qp.py",
            "--quant_model_path", block_ap_model,
            "--model_family", args.net,
            "--wbits", str(args.wbits),
            "--group_size", str(args.group_size),
            "--learning_rate", str(lr),
            "--dataset", args.dataset,
            "--dataset_format", args.dataset_format,
            "--output_dir", e2e_output,
            "--do_train", "True",
            "--pt_context_len", str(args.pt_context_len),
            "--per_device_train_batch_size", str(args.e2e_batch_size),
            "--per_device_eval_batch_size", str(args.e2e_batch_size),
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--logging_steps", "1",
            "--save_strategy", "epoch",
            "--training_strategy", "epochs",
            "--evaluation_strategy", "steps",
            "--eval_steps", "64",
            "--max_train_samples", "4096",
            "--num_train_epochs", str(args.num_train_epochs),
            "--eval_dataset_size", "64",
            "--bf16",
            "--data_seed", str(args.seed),
            "--max_grad_norm", "0.3",
            "--preprocessing_num_workers", "32",
            "--wandb_project", args.wandb_project,
            "--wandb_run_id", run_id,
            "--scheme", args.scheme,
            "--report_to", "wandb",
        ]
        if args.eval_tasks:
            cmd.extend(["--eval_tasks", args.eval_tasks])
        if args.eval_ppl:
            cmd.append("--do_ppl_eval")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"E2E-QP failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"wandb run: {run_name} (id={run_id})")
    print(f"Project: {args.wandb_project}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
