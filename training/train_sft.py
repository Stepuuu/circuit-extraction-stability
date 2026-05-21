#!/usr/bin/env python3
"""
Supervised training entry point.

Aligned with the circuit-sparsity model configuration:
1. GPTConfig carries all circuit-sparsity model features.
2. Every SPARSITY_CONFIGS entry uses afrac=0.25 (activation sparsity).
3. The positional embedding stays trainable (d_pos_emb=32).

Features:
1. AR exact-match validation every eval_interval steps.
2. Loss and eval metrics saved to CSV.
3. Automatic test-set evaluation after training.
4. Automatic visualization.

Dataset seed allocation (train/val/test disjoint):
- train: seed * 10^9 + counter (a billion-sample unique space per seed)
- val:   seed = 888_888_000 + offset
- test:  seed = 999_999_000 + offset

Usage:
    python training/train_sft.py --config dense --bs 512 --seed 42
"""

import argparse
import os
import sys
import json
import csv
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from runtime.inference.gpt import GPT, GPTConfig
from data.logic_task_generator import TrulyRandomDatasetGenerator


# ============================================================
# Configuration
# ============================================================

# Default expansion_factor (can be overridden on the command line)
DEFAULT_EXPANSION_FACTOR = 8  # circuit-sparsity model default (~800M params)
EXPANSION_FACTOR_MLP = 4  # MLP expansion (fixed)

# Sparsity configs, computed from expansion_factor
# L0frac = pfrac / (expansion_factor * expansion_factor_mlp)
# For expansion_factor=8: L0frac = pfrac / 32
SPARSITY_CONFIGS = {
    # Dense: pfrac=None skips sparsification (L0frac=1.0)
    'dense': {
        'pfrac': None,  # Skip apply_topk_ entirely
        'pfrac_anneal': False,
        'afrac': 0.25,  # activation sparsity, keep 25%
    },
    # L0frac = 0.75 → pfrac = 24 (for ef=8) or 12 (for ef=4)
    'sparse_25': {
        'pfrac': None,  # Will be computed based on expansion_factor
        'pfrac_anneal': True,
        'afrac': 0.25,
        '_L0frac_target': 0.75,  # Used to compute pfrac
    },
    # L0frac = 0.50 → pfrac = 16 (for ef=8) or 8 (for ef=4)
    'sparse_50': {
        'pfrac': None,
        'pfrac_anneal': True,
        'afrac': 0.25,
        '_L0frac_target': 0.50,
    },
    # L0frac = 0.25 → pfrac = 8 (for ef=8) or 4 (for ef=4)
    'sparse_75': {
        'pfrac': None,
        'pfrac_anneal': True,
        'afrac': 0.25,
        '_L0frac_target': 0.25,
    },
    # L0frac = 0.10 → pfrac = 3.2 (for ef=8) or 1.6 (for ef=4)
    'sparse_90': {
        'pfrac': None,
        'pfrac_anneal': True,
        'afrac': 0.25,
        '_L0frac_target': 0.10,
    },
}

# Dataset seed allocation
TRAIN_SEED_MULTIPLIER = 1_000_000_000
VAL_SEED_BASE = 888_888_000
TEST_SEED_BASE = 999_999_000

# Task types
TASK_TYPES = [
    'S_and', 'S_or_l', 'S_or_r', 'S_id',
    'T_and_or', 'T_or_and', 'T_and_or_and',
    'T_nested_and', 'T_nested_or'
]


def compute_pfrac_for_config(config_name: str, expansion_factor: int) -> float:
    """Compute pfrac from the expansion factor."""
    cfg = SPARSITY_CONFIGS[config_name]
    if cfg['pfrac'] is not None:
        return cfg['pfrac']
    if '_L0frac_target' not in cfg:
        return None  # Dense config
    L0frac_target = cfg['_L0frac_target']
    # L0frac = pfrac / (expansion_factor * expansion_factor_mlp)
    # pfrac = L0frac * expansion_factor * expansion_factor_mlp
    return L0frac_target * expansion_factor * EXPANSION_FACTOR_MLP


# ============================================================
# Sparsity Functions (from circuit_sparsity/train.py)
# ============================================================

def pfrac_to_L0frac(pfrac, expansion_factor, expansion_factor_mlp, is_embed=False, is_bias=False):
    """Convert pfrac to L0frac (fraction of weights kept alive)"""
    assert not (is_embed and is_bias)
    if is_embed or is_bias:
        return pfrac / expansion_factor
    else:
        return pfrac / (expansion_factor * expansion_factor_mlp)


def apply_topk_(model, pfrac, expansion_factor, expansion_factor_mlp, final_pfrac):
    """Apply top-k sparsification to model parameters.

    Zeros out the smallest weights to achieve target sparsity level.
    Based on circuit_sparsity/train.py apply_topk_ function.
    """
    from functools import partial

    def _lerp(a, b, frac):
        return a + (b - a) * frac

    # For embed/bias layers, pfrac is adjusted via lerp
    # When pfrac == final_pfrac, this returns final_pfrac (no adjustment)
    # When pfrac > final_pfrac, it interpolates
    denom = expansion_factor * expansion_factor_mlp - final_pfrac
    if abs(denom) < 1e-8:
        # Avoid division by zero (happens when final_pfrac == expansion_factor * expansion_factor_mlp)
        # This means L0frac >= 1.0 → no sparsity needed
        return

    _maybe_adjust_pfrac_embbias = lambda x: _lerp(
        final_pfrac,
        expansion_factor,
        (x - final_pfrac) / denom,
    )

    for pn, p in model.named_parameters():
        if len(p.shape) > 1 or "bias" in pn:
            if "bigram_table" in pn:
                continue

            if p is model.transformer.wte.weight:
                L0frac = pfrac_to_L0frac(
                    _maybe_adjust_pfrac_embbias(pfrac),
                    expansion_factor, expansion_factor_mlp,
                    is_embed=True,
                )
            elif p is model.lm_head.weight:
                L0frac = pfrac_to_L0frac(
                    _maybe_adjust_pfrac_embbias(pfrac),
                    expansion_factor, expansion_factor_mlp,
                    is_embed=True,
                )
            elif "bias" in pn:
                L0frac = pfrac_to_L0frac(
                    _maybe_adjust_pfrac_embbias(pfrac),
                    expansion_factor, expansion_factor_mlp,
                    is_bias=True,
                )
            else:
                L0frac = pfrac_to_L0frac(
                    pfrac, expansion_factor, expansion_factor_mlp, is_embed=False
                )

            L0frac = min(L0frac, 1)
            k = int(L0frac * p.numel())

            if k >= p.numel():
                continue  # No sparsification needed

            if k == 0:
                p.data.zero_()
                continue

            # Global top-k by absolute value
            vals, inds = torch.topk(p.data.abs().flatten(), k, sorted=False)

            if len(p.data.shape) == 2:
                mask = torch.ones_like(p.data.flatten(), dtype=torch.bool)
                mask.index_fill_(0, inds, 0)
                mask = mask.view_as(p.data)
                p.data[mask] = 0


def get_current_pfrac(step, total_steps, final_pfrac, pfrac_anneal, expansion_factor):
    """Calculate current pfrac with optional annealing.

    Annealing: start from expansion_factor * expansion_factor_mlp (dense),
    linearly decrease to final_pfrac over the first 20% of training.
    """
    if not pfrac_anneal:
        return final_pfrac

    anneal_steps = int(0.2 * total_steps)
    start_pfrac = expansion_factor * EXPANSION_FACTOR_MLP  # Start dense

    if step >= anneal_steps:
        return final_pfrac

    # Linear interpolation from start_pfrac to final_pfrac
    frac = step / anneal_steps
    return start_pfrac + (final_pfrac - start_pfrac) * frac


# ============================================================
# Data Generation
# ============================================================

def generate_sample(task_type: str, seed: int, n_distractors: int = 2):
    """Generate a single sample."""
    gen = TrulyRandomDatasetGenerator(seed=seed, n_distractors=n_distractors)
    sample = gen.generate_sample(task_type)
    input_text, output_text = sample.to_input_output()
    return input_text, output_text


def create_batch(tokenizer, seeds: List[int], n_ctx: int):
    """Build one batch of training data."""
    import random
    tokens_batch = []

    for seed in seeds:
        rng = random.Random(seed)
        task_type = rng.choice(TASK_TYPES)

        input_text, output_text = generate_sample(task_type, seed)
        text = input_text + output_text
        tokens = tokenizer.encode(text)

        # Pad/truncate
        if len(tokens) > n_ctx:
            tokens = tokens[:n_ctx]
        else:
            tokens = tokens + [0] * (n_ctx - len(tokens))

        tokens_batch.append(tokens)

    return torch.tensor(tokens_batch, dtype=torch.long)


def create_eval_dataset(tokenizer, n_ctx: int, n_samples_per_task: int, seed_base: int):
    """Build the evaluation dataset."""
    eval_data = []
    seed = seed_base

    for task_type in TASK_TYPES:
        for i in range(n_samples_per_task):
            try:
                input_text, output_text = generate_sample(task_type, seed)
                input_tokens = tokenizer.encode(input_text)

                eval_data.append({
                    'task_type': task_type,
                    'input_text': input_text,
                    'output_text': output_text,
                    'input_tokens': input_tokens,
                })
            except:
                pass
            seed += 1

    return eval_data


# ============================================================
# Evaluation Functions
# ============================================================

def extract_first_tactic(text: str) -> str:
    """Extract the first tactic from generated text."""
    if 'state_1' in text:
        text = text.split('state_1')[0]
    elif 'proof complete' in text:
        text = text.split('proof complete')[0]
    elif 'no goals' in text:
        text = text.split('no goals')[0]

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return ""

    if lines[0] == 'state_0_tactic_0:' and len(lines) > 1:
        first_line = lines[1]
    else:
        first_line = lines[0]
        if ':' in first_line and first_line.startswith('state'):
            first_line = first_line.split(':', 1)[1].strip()

    return first_line


def get_expected_tactic(output_text: str) -> str:
    """Extract the tactic from the expected output."""
    lines = output_text.strip().split('\n')
    if len(lines) < 2:
        return output_text.strip()
    return lines[1].strip()


@torch.no_grad()
def evaluate_ar_accuracy(model, tokenizer, eval_data, device, n_ctx, max_samples=None, debug_first_n=0):
    """Evaluate AR exact-match accuracy."""
    model.eval()

    task_correct = {t: 0 for t in TASK_TYPES}
    task_total = {t: 0 for t in TASK_TYPES}

    samples = eval_data[:max_samples] if max_samples else eval_data

    for i, sample in enumerate(samples):
        task_type = sample['task_type']
        input_tokens = sample['input_tokens']
        output_text = sample['output_text']

        # AR generation
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)

        for _ in range(50):  # max new tokens
            if input_ids.shape[1] >= n_ctx:
                break

            output = model(input_ids)
            if isinstance(output, tuple):
                logits = output[0][:, -1, :]
            else:
                logits = output[:, -1, :]

            next_id = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            decoded = tokenizer.decode(input_ids[0].tolist())
            if "proof complete" in decoded or "no goals" in decoded:
                break

        # Extract and compare
        generated_ids = input_ids[0, len(input_tokens):].tolist()
        generated_text = tokenizer.decode(generated_ids)

        pred_tactic = extract_first_tactic(generated_text)
        expected_tactic = get_expected_tactic(output_text)

        # Debug output for first few samples
        if i < debug_first_n:
            print(f"\n  [Debug Sample {i}] Task: {task_type}")
            print(f"    Expected: '{expected_tactic}'")
            print(f"    Generated: '{generated_text[:100]}...'")
            print(f"    Extracted: '{pred_tactic}'")
            print(f"    Match: {pred_tactic == expected_tactic}")

        task_total[task_type] += 1
        if pred_tactic == expected_tactic:
            task_correct[task_type] += 1

    model.train()

    # Calculate accuracies
    task_acc = {t: (task_correct[t] / task_total[t] * 100 if task_total[t] > 0 else 0)
                for t in TASK_TYPES}
    overall_correct = sum(task_correct.values())
    overall_total = sum(task_total.values())
    overall_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0

    return {
        'overall_accuracy': overall_acc,
        'overall_correct': overall_correct,
        'overall_total': overall_total,
        'task_accuracy': task_acc,
    }


# ============================================================
# Training Metrics Logger
# ============================================================

class MetricsLogger:
    """Log training and validation metrics."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CSV files
        self.train_csv = output_dir / "train_metrics.csv"
        self.eval_csv = output_dir / "eval_metrics.csv"

        # Initialize CSV files
        with open(self.train_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss', 'lr', 'tokens_seen', 'time_elapsed'])

        with open(self.eval_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['step', 'overall_acc'] + [f'acc_{t}' for t in TASK_TYPES]
            writer.writerow(header)

        self.start_time = time.time()

    def log_train(self, step: int, loss: float, lr: float, tokens_seen: int):
        elapsed = time.time() - self.start_time
        with open(self.train_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, f'{loss:.6f}', f'{lr:.8f}', tokens_seen, f'{elapsed:.1f}'])

    def log_eval(self, step: int, eval_results: Dict):
        with open(self.eval_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [step, f"{eval_results['overall_accuracy']:.2f}"]
            for t in TASK_TYPES:
                row.append(f"{eval_results['task_accuracy'].get(t, 0):.2f}")
            writer.writerow(row)


# ============================================================
# Main Training Function
# ============================================================

def run_experiment(
    sparsity_config: str = "dense",
    batch_size: int = 512,
    seed: int = 42,
    tokenizer_path: str = "artifacts/tokenizer/trained/tokenizer.model",
    output_base: str = "artifacts/results",
    total_tokens: int = 1_000_000_000,
    n_ctx: int = 128,
    n_layer: int = 8,
    vocab_size: int = 2048,
    expansion_factor: int = 8,  # 8 = large model (~800M), matching the circuit-sparsity model
    eval_interval: int = 500,  # validate every N steps
    eval_samples: int = 50,  # samples per task during validation
    test_samples: int = 100,  # samples per task for the final test
    log_interval: int = 50,  # log every N steps
    save_interval: int = 2000,  # save every N steps
    device: str = "cuda",
):
    """Run training with periodic evaluation"""

    # Get sparsity config
    if sparsity_config not in SPARSITY_CONFIGS:
        raise ValueError(f"Unknown sparsity config: {sparsity_config}")
    sparse_cfg = SPARSITY_CONFIGS[sparsity_config].copy()

    # Compute pfrac based on expansion_factor
    sparse_cfg['pfrac'] = compute_pfrac_for_config(sparsity_config, expansion_factor)

    # Calculate L0frac based on expansion_factor
    expansion_product = expansion_factor * EXPANSION_FACTOR_MLP
    if sparse_cfg['pfrac'] is not None:
        L0frac = sparse_cfg['pfrac'] / expansion_product
    else:
        L0frac = 1.0

    # Setup paths
    exp_name = f"{sparsity_config}_bs{batch_size}_seed{seed}"
    # output_base can be either:
    # 1. "artifacts/results" (base dir) -> append checkpoints/exp_name
    # 2. "artifacts/results/checkpoints/exp_name" (full path) -> use directly
    if output_base.endswith(exp_name) or "checkpoints/" in output_base:
        # New style: output_base already includes the experiment subdir
        output_dir = Path(output_base) / exp_name
    else:
        # Old style: add checkpoints/exp_name
        output_dir = Path(output_base) / "checkpoints" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print config
    print("=" * 70)
    print(f"Experiment: {exp_name}")
    print("=" * 70)
    print(f"Sparsity: {sparsity_config}, pfrac={sparse_cfg['pfrac']}")
    print(f"Expansion factor: {expansion_factor}")
    if sparse_cfg['pfrac'] is not None:
        print(f"  L0frac: {L0frac:.4f}")
    else:
        print(f"  L0frac: 1.0 (dense)")
    print(f"Activation sparsity (afrac): {sparse_cfg['afrac']}")
    print(f"Batch size: {batch_size}, n_ctx: {n_ctx}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Eval interval: {eval_interval} steps, {eval_samples} samples/task")
    print(f"Output: {output_dir}")
    print()

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    # Calculate steps
    total_steps = total_tokens // (batch_size * n_ctx)
    print(f"Total steps: {total_steps:,}")

    # Build the model config, fully mirroring the circuit-sparsity model
    base_d_model = 256
    base_n_head = 8
    base_d_head = 32
    d_head = 16

    d_model = int(base_d_model * expansion_factor)
    n_head = int(base_n_head * expansion_factor * base_d_head) // d_head
    d_mlp = int(base_d_model * 4 * expansion_factor)

    # afrac_loctypes: the full set of activation-sparsity locations
    afrac_loctypes = "attn_in,attn_out,mlp_in,mlp_out,mlp_neuron,attn_v,attn_k,attn_q"

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
        rms_norm=True,
        activation_type="gelu",
        residual_activation_type="identity",
        enable_bigram_table=True,
        learnable_bigram_table=True,
        # === Fully aligned with the circuit-sparsity model ===
        d_pos_emb=32,  # concatenated positional encoding
        sink=True,                     # attention sink
        flash=True,  # Flash Attention (required by the sink)
        afrac=sparse_cfg['afrac'],  # activation sparsity
        afrac_loctypes=afrac_loctypes,  # activation-sparsity locations
        tied_unembed=False,  # untied embedding
    )

    model = GPT(model_config).to(device)

    # Note: with d_pos_emb=32, wpe is a learnable (block_size, 32) positional embedding
    # Keep wpe trainable (the circuit-sparsity model also uses learned pos emb)
    # The old wpe freezing logic no longer applies

    if hasattr(model, 'bigram_table'):
        model.bigram_table.data = torch.rand_like(model.bigram_table.data) * 0.02

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  d_model={d_model}, n_head={n_head}, d_mlp={d_mlp}")
    print(f"  d_pos_emb=32 (concatenated position embedding)")
    print(f"  sink=True, flash=True")
    print(f"  tied_unembed=False")

    # Optimizer - lr=3e-4 for large model
    lr = 3e-4
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda')

    # Initialize metrics logger
    logger = MetricsLogger(output_dir)

    # Create validation dataset
    print("Creating validation dataset...")
    val_data = create_eval_dataset(tokenizer, n_ctx, eval_samples, VAL_SEED_BASE)
    print(f"Validation samples: {len(val_data)}")

    # Save config
    exp_config = {
        'sparsity_config': sparsity_config,
        'pfrac': sparse_cfg['pfrac'],
        'L0frac': L0frac,
        'afrac': sparse_cfg['afrac'],
        'afrac_loctypes': afrac_loctypes,
        'expansion_factor': expansion_factor,
        'batch_size': batch_size,
        'seed': seed,
        'total_tokens': total_tokens,
        'total_steps': total_steps,
        'n_ctx': n_ctx,
        'n_layer': n_layer,
        'd_model': d_model,
        'n_head': n_head,
        'd_mlp': d_mlp,
        'd_pos_emb': 32,
        'sink': True,
        'tied_unembed': False,
        'vocab_size': vocab_size,
        'eval_interval': eval_interval,
        'eval_samples': eval_samples,
        'test_samples': test_samples,
        'timestamp': datetime.now().isoformat(),
        'script': 'training/train_sft.py',
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(exp_config, f, indent=2)

    # Training loop
    print("\nStarting training...")
    model.train()

    sample_counter = 0
    best_acc = -1  # Start at -1 so first eval always saves
    pbar = tqdm(total=total_steps, desc="Training")

    for step in range(1, total_steps + 1):
        # Generate batch
        seeds = [seed * TRAIN_SEED_MULTIPLIER + sample_counter + i for i in range(batch_size)]
        sample_counter += batch_size

        batch = create_batch(tokenizer, seeds, n_ctx).to(device)

        # LR schedule (warmup then constant, like successful train_unified.py)
        warmup_steps = 1000  # Fixed 1000 steps warmup (proven to work)
        if step < warmup_steps:
            current_lr = lr * step / warmup_steps
        else:
            # Constant LR after warmup (no decay for simpler training)
            current_lr = lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Forward pass
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            output = model(batch[:, :-1])
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
                ignore_index=0,  # ignore padding
            )

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Apply sparsity mask (if not dense)
        if sparse_cfg['pfrac'] is not None:
            current_pfrac = get_current_pfrac(
                step, total_steps, sparse_cfg['pfrac'], sparse_cfg['pfrac_anneal'],
                expansion_factor
            )
            apply_topk_(
                model, current_pfrac,
                expansion_factor, EXPANSION_FACTOR_MLP,
                final_pfrac=sparse_cfg['pfrac'],
            )

        # Logging
        tokens_seen = step * batch_size * n_ctx

        if step % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
            })
            logger.log_train(step, loss.item(), current_lr, tokens_seen)

        # Evaluation
        if step % eval_interval == 0 or step == total_steps:
            # Debug: print first 3 samples at final step
            debug_n = 3 if step == total_steps else 0
            eval_results = evaluate_ar_accuracy(
                model, tokenizer, val_data, device, n_ctx, debug_first_n=debug_n
            )
            logger.log_eval(step, eval_results)

            print(f"\n[Step {step}] Val AR Accuracy: {eval_results['overall_accuracy']:.2f}%")

            # Save best model
            if eval_results['overall_accuracy'] > best_acc:
                best_acc = eval_results['overall_accuracy']
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                print(f"  New best! Saved to best_model.pt")

        # Save checkpoint
        if step % save_interval == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, output_dir / f"checkpoint_step{step}.pt")

        pbar.update(1)

    pbar.close()

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    print(f"\nTraining complete! Final model saved.")

    # ===== Final Test Evaluation =====
    print("\n" + "=" * 70)
    print("Running final test evaluation...")
    print("=" * 70)

    # Load best model for testing
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))

    # Create test dataset
    test_data = create_eval_dataset(tokenizer, n_ctx, test_samples, TEST_SEED_BASE)
    print(f"Test samples: {len(test_data)}")

    test_results = evaluate_ar_accuracy(model, tokenizer, test_data, device, n_ctx)

    # Save test results
    test_results_full = {
        'overall_accuracy': test_results['overall_accuracy'],
        'overall_correct': test_results['overall_correct'],
        'overall_total': test_results['overall_total'],
        'task_accuracy': test_results['task_accuracy'],
        'test_samples_per_task': test_samples,
        'model': 'best_model.pt',
    }

    with open(output_dir / "test_results.json", 'w') as f:
        json.dump(test_results_full, f, indent=2)

    # Print test results
    print(f"\nTest Results (AR Exact Match):")
    print(f"  Overall: {test_results['overall_accuracy']:.2f}%")
    print(f"  Per-task:")
    for task, acc in test_results['task_accuracy'].items():
        print(f"    {task}: {acc:.2f}%")

    print(f"\nResults saved to: {output_dir}")
    print("  - config.json")
    print("  - train_metrics.csv")
    print("  - eval_metrics.csv")
    print("  - test_results.json")
    print("  - best_model.pt")
    print("  - final_model.pt")
    print("\nExperiment complete")

    return test_results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Training with circuit-sparsity-model config")

    parser.add_argument("--config", type=str, default="dense",
                        choices=list(SPARSITY_CONFIGS.keys()))
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--n-ctx", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--expansion-factor", type=int, default=8,
                        help="Model expansion factor (8=~800M params, same as circuit-sparsity-model)")
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/trained/tokenizer.model")
    parser.add_argument("--output", default="artifacts/results")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    run_experiment(
        sparsity_config=args.config,
        batch_size=args.bs,
        seed=args.seed,
        tokenizer_path=args.tokenizer,
        output_base=args.output,
        total_tokens=args.total_tokens,
        n_ctx=args.n_ctx,
        n_layer=args.n_layer,
        vocab_size=args.vocab_size,
        expansion_factor=args.expansion_factor,
        eval_interval=args.eval_interval,
        eval_samples=args.eval_samples,
        test_samples=args.test_samples,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=args.device,
    )


if __name__ == "__main__":
    main()
