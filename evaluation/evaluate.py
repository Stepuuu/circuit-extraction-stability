#!/usr/bin/env python3
"""
Task-level evaluation script.

Strict exact-match autoregressive (AR) evaluation. The test set uses a seed
space fully disjoint from training/validation (TEST_SEED_BASE = 999_999_000).

Metrics:
1. AR exact match: the first generated tactic must exactly match the ground truth.
2. Per-task accuracy for every task type.

Usage:
    # Evaluate a single model
    python evaluation/evaluate.py --model artifacts/results/checkpoints/dense_bs512_seed42

    # Evaluate all models
    python evaluation/evaluate.py --all

    # Custom number of test samples
    python evaluation/evaluate.py --all --n-samples 500
"""

import argparse
import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import sentencepiece as spm
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from runtime.inference.gpt import GPT, GPTConfig
from data.family_generator import generate_sample as generate_family_sample

# Test-set seeds (fully disjoint from train/val)
TEST_SEED_BASE = 999_999_000

# Task types
TASK_TYPES = [
    'S_and', 'S_or_l', 'S_or_r', 'S_id',
    'T_and_or', 'T_or_and', 'T_and_or_and',
    'T_nested_and', 'T_nested_or'
]


def extract_first_tactic(text):
    """Extract the first tactic from generated text (for strict matching)."""
    # Truncate before state_1 or proof completion
    if 'state_1' in text:
        text = text.split('state_1')[0]
    elif 'proof complete' in text:
        text = text.split('proof complete')[0]
    elif 'no goals' in text:
        text = text.split('no goals')[0]

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return ""

    # Handle the state_0_tactic_0: prefix
    if lines[0] == 'state_0_tactic_0:' and len(lines) > 1:
        first_line = lines[1]
    else:
        first_line = lines[0]
        if ':' in first_line and first_line.startswith('state'):
            first_line = first_line.split(':', 1)[1].strip()

    return first_line


def get_expected_tactic(output_text):
    """Extract the tactic from the expected output."""
    lines = output_text.strip().split('\n')
    # Format: state_0_tactic_0:\nexact xxx\nstate_1:...
    if len(lines) < 2:
        return output_text.strip()
    tactic_line = lines[1]
    return tactic_line.strip()


def generate_test_sample(task_type, seed, n_distractors=2):
    """Generate a test sample of the given task type."""
    input_text, output_text = generate_family_sample(task_type, seed, n_distractors=n_distractors)
    return {
        'input_text': input_text,
        'output_text': output_text,
        'task_type': task_type,
    }


def create_test_dataset(n_samples_per_task=100, seed_base=TEST_SEED_BASE):
    """Build the test dataset (organized by task type)."""
    test_data = {task: [] for task in TASK_TYPES}

    for task_type in TASK_TYPES:
        seed = seed_base
        collected = 0
        attempts = 0
        max_attempts = n_samples_per_task * 3

        while collected < n_samples_per_task and attempts < max_attempts:
            try:
                sample = generate_test_sample(task_type, seed + attempts)
                test_data[task_type].append(sample)
                collected += 1
            except Exception:
                pass
            attempts += 1
            seed_base += 1  # increment globally so different tasks use different seeds

    return test_data


@torch.no_grad()
def generate_ar(model, tokenizer, device, prompt_text, max_new_tokens=80, n_ctx=256):
    """Autoregressive generation (greedy decoding)."""
    prompt_ids = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= n_ctx:
            break

        output = model(input_ids)
        if isinstance(output, tuple):
            logits = output[0][:, -1, :]
        else:
            logits = output[:, -1, :]

        next_id = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)

        # Check whether generation is complete
        decoded = tokenizer.decode(input_ids[0].tolist())
        if "proof complete" in decoded or "no goals" in decoded:
            break

    generated_ids = input_ids[0, len(prompt_ids):].tolist()
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


def load_model_from_checkpoint(checkpoint_dir, device="cuda"):
    """Load a model from a checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)

    # Load the experiment config
    config_file = checkpoint_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")

    with open(config_file, 'r') as f:
        exp_config = json.load(f)

    # Locate checkpoint files
    # Saved checkpoints may be named model_step_xxx.pt or similar
    ckpt_files = list(checkpoint_path.glob("*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    # Prefer best or final checkpoints, otherwise use the latest
    best_ckpt = checkpoint_path / "best.pt"
    final_ckpt = checkpoint_path / "final.pt"

    if best_ckpt.exists():
        ckpt_file = best_ckpt
    elif final_ckpt.exists():
        ckpt_file = final_ckpt
    else:
        # Pick the one with the largest step
        ckpt_file = sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1]

    print(f"Loading checkpoint: {ckpt_file.name}")

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)

    # Derive the actual model config from expansion_factor and the base config
    # smol_model_config: d_model=256, n_head=8, d_head=32
    # expansion_factor=4: d_model=1024, n_head=64, d_head=16, d_mlp=4096
    expansion_factor = exp_config.get('expansion_factor', 4)
    n_layer = exp_config.get('n_layer', 8)
    vocab_size = exp_config.get('vocab_size', 2048)
    n_ctx = exp_config.get('n_ctx', 256)

    # Compute actual dimensions
    base_d_model = 256
    base_n_head = 8
    base_d_head = 32
    d_head = 16  # fixed

    d_model = int(base_d_model * expansion_factor)
    n_head = int(base_n_head * expansion_factor * base_d_head) // d_head
    d_mlp = int(base_d_model * 4 * expansion_factor)

    model_config = GPTConfig(
        block_size=n_ctx,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        d_head=d_head,
        d_model=d_model,
        d_mlp=d_mlp,
        dropout=0.0,
        bias=True,
        flash=False,
        rms_norm=True,
        activation_type="gelu",
        residual_activation_type="identity",
        enable_bigram_table=True,
        learnable_bigram_table=True,
    )

    model = GPT(model_config)

    # Handle the state_dict (it may carry a module. prefix)
    state_dict = checkpoint if isinstance(checkpoint, dict) and 'model' not in checkpoint else checkpoint.get('model', checkpoint)

    # If this is a full checkpoint dict
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, exp_config


def evaluate_model(model, tokenizer, test_data, device="cuda", n_ctx=256, max_samples_per_task=100):
    """Evaluate a model with strict exact-match AR evaluation."""
    task_results = {}
    overall_correct = 0
    overall_total = 0

    for task_type in TASK_TYPES:
        samples = test_data[task_type][:max_samples_per_task]
        correct = 0
        total = 0

        for sample in samples:
            input_text = sample['input_text']
            output_text = sample['output_text']

            # AR generation
            generated = generate_ar(model, tokenizer, device, input_text, n_ctx=n_ctx)

            # Extract tactics
            pred_tactic = extract_first_tactic(generated)
            expected_tactic = get_expected_tactic(output_text)

            # Strict match
            if pred_tactic == expected_tactic:
                correct += 1
            total += 1

        acc = correct / total * 100 if total > 0 else 0
        task_results[task_type] = {
            'correct': correct,
            'total': total,
            'accuracy': acc
        }
        overall_correct += correct
        overall_total += total

    overall_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0

    return {
        'task_results': task_results,
        'overall_accuracy': overall_acc,
        'overall_correct': overall_correct,
        'overall_total': overall_total,
    }


def evaluate_single(
    model_dir: str,
    tokenizer_path: str,
    n_samples_per_task: int = 100,
    device: str = "cuda",
):
    """Evaluate a single model."""
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    print(f"Creating test dataset ({n_samples_per_task} samples per task)...")
    test_data = create_test_dataset(n_samples_per_task)

    print(f"Loading model: {model_dir}")
    model, exp_config = load_model_from_checkpoint(model_dir, device)
    n_ctx = exp_config.get('n_ctx', 256)

    print("Evaluating...")
    results = evaluate_model(
        model, tokenizer, test_data,
        device=device, n_ctx=n_ctx, max_samples_per_task=n_samples_per_task
    )
    results['config'] = exp_config

    return results


def evaluate_all(
    results_dir: str = "artifacts/results/checkpoints",
    tokenizer_path: str = "artifacts/tokenizer/trained/tokenizer.model",
    output_file: str = "artifacts/results/evaluation_report.json",
    n_samples_per_task: int = 100,
    device: str = None,
):
    """Evaluate all models in batch."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return {}

    # Load the tokenizer
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    # Build the test dataset (shared across models)
    print(f"Creating test dataset ({n_samples_per_task} samples per task, {len(TASK_TYPES)} tasks)...")
    test_data = create_test_dataset(n_samples_per_task)
    total_samples = sum(len(samples) for samples in test_data.values())
    print(f"Total test samples: {total_samples}")

    # Find all experiment directories
    exp_dirs = [d for d in results_path.iterdir() if d.is_dir() and (d / "config.json").exists()]
    print(f"Found {len(exp_dirs)} experiments")

    all_results = {}

    for exp_dir in tqdm(exp_dirs, desc="Evaluating"):
        exp_name = exp_dir.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")

        try:
            model, exp_config = load_model_from_checkpoint(str(exp_dir), device)
            n_ctx = exp_config.get('n_ctx', 256)

            results = evaluate_model(
                model, tokenizer, test_data,
                device=device, n_ctx=n_ctx, max_samples_per_task=n_samples_per_task
            )
            results['config'] = exp_config
            all_results[exp_name] = results

            print(f"  Overall AR Accuracy: {results['overall_accuracy']:.2f}%")
            print(f"  Per-task:")
            for task, task_res in results['task_results'].items():
                print(f"    {task}: {task_res['accuracy']:.1f}%")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {'error': str(e)}

        # Free GPU memory
        if 'model' in dir():
            del model
        torch.cuda.empty_cache()

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Print the summary
    print_summary(all_results)

    return all_results


def print_summary(results: dict):
    """Print the evaluation summary table."""
    print("\n" + "=" * 100)
    print("Evaluation Summary (AR Exact Match)")
    print("=" * 100)

    # Group by sparsity config
    grouped = defaultdict(list)
    for exp_name, result in results.items():
        if 'error' in result:
            continue
        # Parse the experiment name: {config}_bs{bs}_seed{seed}
        parts = exp_name.split('_bs')
        if len(parts) == 2:
            config = parts[0]
            grouped[config].append((exp_name, result))

    # Print the header
    print(f"\n{'Config':<15} | {'Exp Name':<35} | {'Overall':>10} | " +
          " | ".join(f"{t[:6]:>6}" for t in TASK_TYPES))
    print("-" * 120)

    for config in ['dense', 'sparse_25', 'sparse_50', 'sparse_75', 'sparse_90', 'default']:
        if config not in grouped:
            continue

        for exp_name, result in sorted(grouped[config]):
            task_accs = [result['task_results'].get(t, {}).get('accuracy', 0) for t in TASK_TYPES]
            task_str = " | ".join(f"{acc:>6.1f}" for acc in task_accs)
            print(f"{config:<15} | {exp_name:<35} | {result['overall_accuracy']:>9.2f}% | {task_str}")

        # Compute means
        if len(grouped[config]) > 1:
            avg_overall = sum(r['overall_accuracy'] for _, r in grouped[config]) / len(grouped[config])
            avg_tasks = []
            for t in TASK_TYPES:
                task_avg = sum(r['task_results'].get(t, {}).get('accuracy', 0) for _, r in grouped[config]) / len(grouped[config])
                avg_tasks.append(task_avg)
            task_str = " | ".join(f"{acc:>6.1f}" for acc in avg_tasks)
            print(f"{'[Average]':<15} | {'':<35} | {avg_overall:>9.2f}% | {task_str}")

        print("-" * 120)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models with strict AR exact match")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to single model checkpoint directory")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all models in results directory")
    parser.add_argument("--results-dir", default="artifacts/results/checkpoints",
                        help="Results directory")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/trained/tokenizer.model",
                        help="Path to tokenizer")
    parser.add_argument("--output", default="artifacts/results/evaluation_report.json",
                        help="Output file for evaluation report")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples per task type")
    parser.add_argument("--device", default=None,
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    if args.all:
        evaluate_all(
            results_dir=args.results_dir,
            tokenizer_path=args.tokenizer,
            output_file=args.output,
            n_samples_per_task=args.n_samples,
            device=args.device,
        )
    elif args.model:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        results = evaluate_single(
            model_dir=args.model,
            tokenizer_path=args.tokenizer,
            n_samples_per_task=args.n_samples,
            device=device,
        )

        print("\n" + "=" * 60)
        print("Evaluation Results (AR Exact Match)")
        print("=" * 60)
        print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"Correct: {results['overall_correct']} / {results['overall_total']}")
        print("\nPer-task:")
        for task, task_res in results['task_results'].items():
            print(f"  {task:<15}: {task_res['accuracy']:>6.2f}% ({task_res['correct']}/{task_res['total']})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
