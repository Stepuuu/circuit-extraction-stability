#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training entry point.

Refines an SFT checkpoint with reinforcement learning to improve accuracy on
compositional tasks.

Core idea:
  1. Sample G candidate responses per prompt.
  2. Score each response with a verifier (correct=1, wrong=0).
  3. Update the policy using group-relative advantages.

Usage:
    python training/train_grpo.py \
        --model-type custom \
        --model-path <checkpoint> \
        --config-path <config> \
        --output-dir artifacts/results/grpo_run
"""

import argparse
import json
import os
import sys
import copy
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath("."))

from data.family_generator import generate_sample as generate_family_sample


# ============================================================
# Configuration
# ============================================================

TASK_TYPES = [
    'S_and', 'S_or_l', 'S_or_r', 'S_id',
    'T_and_or', 'T_or_and', 'T_and_or_and',
    'T_nested_and', 'T_nested_or'
]

SIMPLE_TASKS = ['S_and', 'S_or_l', 'S_or_r', 'S_id']
COMPOSITION_TASKS = ['T_and_or', 'T_or_and', 'T_and_or_and', 'T_nested_and', 'T_nested_or']


@dataclass
class GRPOConfig:
    """GRPO training configuration."""
    # Sampling
    num_generations: int = 8  # G responses per prompt
    temperature: float = 1.0  # sampling temperature
    max_new_tokens: int = 50  # max new tokens (block_size=128, input ~70, leave headroom)
    
    # Optimization
    learning_rate: float = 1e-5  # smaller LR for the RL stage
    beta: float = 0.1  # KL penalty coefficient
    clip_eps: float = 0.2  # PPO-style clipping (optional)
    
    # Training
    batch_size: int = 8  # prompts per batch
    gradient_accumulation: int = 4
    epochs: int = 10  # matches the reported experiments
    warmup_ratio: float = 0.1
    
    # Reward
    correct_reward: float = 1.0
    wrong_reward: float = 0.0
    format_penalty: float = -0.1  # penalty for malformed output
    
    # Misc
    seed: int = 42
    eval_interval_early: int = 5  # eval every 5 steps within the first 100 steps
    eval_interval_later: int = 50  # eval every 50 steps afterwards
    eval_early_threshold: int = 100  # dense eval within the first 100 steps
    save_interval: int = 500


# ============================================================
# Data generation
# ============================================================

def generate_sample(task_type: str, seed: int, n_distractors: int = 2) -> Tuple[str, str]:
    """Generate one sample; LOGIC_GENERATOR_FAMILY switches the narrow/broad family."""
    return generate_family_sample(task_type, seed, n_distractors=n_distractors)


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
    """Extract the expected tactic from the ground truth."""
    lines = output_text.strip().split('\n')
    if len(lines) < 2:
        return output_text.strip()
    return lines[1].strip()


class RLDataset(Dataset):
    """GRPO training dataset."""
    
    def __init__(self, task_types: List[str], num_samples: int, seed_base: int = 0):
        self.samples = []
        seed = seed_base
        
        for _ in range(num_samples):
            for task_type in task_types:
                try:
                    input_text, output_text = generate_sample(task_type, seed)
                    expected_tactic = get_expected_tactic(output_text)
                    self.samples.append({
                        'task_type': task_type,
                        'input_text': input_text,
                        'output_text': output_text,
                        'expected_tactic': expected_tactic,
                    })
                except Exception as e:
                    pass
                seed += 1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# Reward function
# ============================================================

def compute_reward(
    generated_text: str,
    expected_tactic: str,
    config: GRPOConfig
) -> float:
    """Compute the reward for a single generation."""
    predicted_tactic = extract_first_tactic(generated_text)
    
    if not predicted_tactic:
        # No valid content generated
        return config.format_penalty
    
    if predicted_tactic == expected_tactic:
        return config.correct_reward
    else:
        return config.wrong_reward


def compute_rewards_batch(
    generated_texts: List[str],
    expected_tactic: str,
    config: GRPOConfig
) -> Tuple[List[float], List[float]]:
    """Compute rewards and advantages for a group of generations.
    
    Returns:
        rewards: list of raw rewards
        advantages: group-normalized advantages
    """
    rewards = [compute_reward(t, expected_tactic, config) for t in generated_texts]
    
    # GRPO: normalize within the group
    mean_r = np.mean(rewards)
    std_r = np.std(rewards) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]
    
    return rewards, advantages


# ============================================================
# Model loading
# ============================================================

def load_model_gpt2(model_path: str, device: str):
    """Load a GPT-2 model."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_model_qwen(model_path: str, device: str):
    """Load a Qwen model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    return model, tokenizer


def load_model_custom(model_path: str, config_path: str, device: str):
    """Load the custom circuit-sparsity model."""
    import sentencepiece as spm
    from runtime.inference.gpt import GPT, GPTConfig
    
    # Load config
    with open(config_path, 'r') as f:
        exp_config = json.load(f)
    
    # Build model config - Match SFT training config exactly
    ablation_config = exp_config.get('ablation_config', {})
    gpt_config = GPTConfig(
        block_size=exp_config.get('n_ctx', 128),
        vocab_size=exp_config.get('vocab_size', 2048),
        n_layer=exp_config.get('n_layer', 8),
        n_head=exp_config.get('n_head', 128),
        d_head=16,
        d_model=exp_config.get('d_model', 2048),
        d_mlp=exp_config.get('d_mlp', 8192),
        dropout=0.0,
        bias=True,
        rms_norm=ablation_config.get('rms_norm', 'False') == 'True',  # Read from config!
        flash=ablation_config.get('flash', 'True') == 'True',
        enable_bigram_table=True,
        learnable_bigram_table=True,
        tied_unembed=ablation_config.get('tied_unembed', 'False') == 'True',
    )
    
    model = GPT(gpt_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load("artifacts/tokenizer/trained/tokenizer.model")
    
    return model, tokenizer


# ============================================================
# GRPO training
# ============================================================

@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    input_text: str,
    num_generations: int,
    config: GRPOConfig,
    device: str,
    model_type: str = "gpt2",
) -> Tuple[List[str], List[torch.Tensor]]:
    """Generate multiple candidate responses.
    
    Returns:
        responses: list of generated texts
        log_probs: log probability of each response
    """
    if model_type in ["gpt2", "qwen"]:
        # HuggingFace models
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_len = inputs['input_ids'].shape[1]
        
        responses = []
        log_probs_list = []
        
        for _ in range(num_generations):
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            generated_ids = outputs.sequences[0, input_len:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response_text)
            
            # Compute log probs for the generated sequence
            # (simplified version, proper implementation needs more care)
            log_probs = torch.zeros(1, device=device)
            if outputs.scores:
                for i, score in enumerate(outputs.scores):
                    if i < len(generated_ids):
                        token_id = generated_ids[i]
                        log_prob = F.log_softmax(score[0], dim=-1)[token_id]
                        log_probs = log_probs + log_prob
            log_probs_list.append(log_probs)
        
        return responses, log_probs_list
    
    else:
        # Custom model
        input_ids = tokenizer.encode(input_text)
        input_ids = torch.tensor([input_ids], device=device)
        input_len = input_ids.shape[1]
        
        # Get block_size from model config
        block_size = model.config.block_size

        # Encode stop strings once for efficient checking
        stop_strings = ["proof complete", "no goals"]

        responses = []
        log_probs_list = []

        for _ in range(num_generations):
            generated = input_ids.clone()
            log_probs = torch.zeros(1, device=device)

            for _ in range(config.max_new_tokens):
                # Safety check: stop if we're approaching block_size limit
                if generated.shape[1] >= block_size - 1:
                    break
                
                output = model(generated)
                if isinstance(output, tuple):
                    logits = output[0][:, -1, :]
                else:
                    logits = output[:, -1, :]

                # Sample
                probs = F.softmax(logits / config.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Log prob (use unscaled logits for correct log_prob)
                log_prob = F.log_softmax(logits, dim=-1).gather(1, next_token)
                log_probs = log_probs + log_prob.squeeze()

                generated = torch.cat([generated, next_token], dim=1)

                # Stop conditions — only decode response part
                resp_text = tokenizer.decode(generated[0, input_len:].tolist())
                if any(s in resp_text for s in stop_strings):
                    break

            response_ids = generated[0, input_len:].tolist()
            response_text = tokenizer.decode(response_ids)
            responses.append(response_text)
            log_probs_list.append(log_probs)

        return responses, log_probs_list


def _compute_log_probs_and_kl(
    model,
    ref_model,
    input_ids: torch.Tensor,
    input_len: int,
    config: GRPOConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the summed log-prob of response tokens and the per-token KL divergence.

    Args:
        input_ids: [1, seq_len] full sequence (prompt + response)
        input_len: number of prompt tokens
    Returns:
        log_prob_sum: scalar, summed log pi(y|x) over response tokens
        kl: scalar, mean KL(pi || pi_ref) over response tokens
    """
    # Forward: feed [0:-1], predict [1:]
    outputs = model(input_ids[:, :-1])
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    elif isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs

    # Response part only: logits[input_len-1:] predict tokens[input_len:]
    resp_logits = logits[:, input_len - 1:, :]
    resp_targets = input_ids[:, input_len:]
    n_resp = resp_targets.shape[1]

    if n_resp == 0:
        zero = torch.tensor(0.0, device=input_ids.device)
        return zero, zero

    # Per-token log prob
    log_probs = F.log_softmax(resp_logits, dim=-1)  # [1, n_resp, vocab]
    token_log_probs = log_probs.gather(2, resp_targets.unsqueeze(-1)).squeeze(-1)  # [1, n_resp]
    log_prob_sum = token_log_probs.sum()

    # KL penalty (only on response tokens)
    kl = torch.tensor(0.0, device=input_ids.device)
    if ref_model is not None and config.beta > 0:
        with torch.no_grad():
            ref_out = ref_model(input_ids[:, :-1])
            if hasattr(ref_out, 'logits'):
                ref_logits = ref_out.logits
            elif isinstance(ref_out, tuple):
                ref_logits = ref_out[0]
            else:
                ref_logits = ref_out
            ref_resp_logits = ref_logits[:, input_len - 1:, :]

        # KL(π || π_ref) = Σ π(x) * [log π(x) - log π_ref(x)]
        ref_log_probs = F.log_softmax(ref_resp_logits, dim=-1)
        probs = F.softmax(resp_logits, dim=-1)
        kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1).mean()

    return log_prob_sum, kl


def compute_policy_loss(
    model,
    tokenizer,
    input_text: str,
    responses: List[str],
    advantages: List[float],
    ref_model,
    config: GRPOConfig,
    device: str,
    model_type: str = "gpt2",
) -> torch.Tensor | None:
    """Compute the GRPO policy loss.

    GRPO loss = - Σ_i [ log π(y_i|x) * A_i ] / G  +  β * KL(π || π_ref)

    where A_i is the group-normalized advantage.
    """
    # Get block_size for length checking
    block_size = model.config.block_size if model_type == "custom" else None
    
    total_pg_loss = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    valid_count = 0

    for response, advantage in zip(responses, advantages):
        full_text = input_text + response

        if model_type in ["gpt2", "qwen"]:
            input_ids = tokenizer(full_text, return_tensors="pt")['input_ids'].to(device)
            input_len = len(tokenizer(input_text)['input_ids'])
        else:
            tokens = tokenizer.encode(full_text)
            input_ids = torch.tensor([tokens], device=device)
            input_len = len(tokenizer.encode(input_text))

        # Skip if sequence is too long for the model
        if block_size and input_ids.shape[1] > block_size:
            continue
        
        log_prob_sum, kl = _compute_log_probs_and_kl(
            model, ref_model, input_ids, input_len, config
        )

        # Policy gradient: -log_prob * advantage (maximize expected advantage)
        total_pg_loss = total_pg_loss - log_prob_sum * advantage
        total_kl = total_kl + kl
        valid_count += 1

    # If all responses were too long, return zero loss
    if valid_count == 0:
        return None
    
    return total_pg_loss / valid_count + config.beta * total_kl / valid_count


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    eval_data: List[dict],
    config: GRPOConfig,
    device: str,
    model_type: str = "gpt2",
) -> Dict:
    """Evaluate model performance."""
    model.eval()
    
    task_correct = {t: 0 for t in TASK_TYPES}
    task_total = {t: 0 for t in TASK_TYPES}
    
    for sample in tqdm(eval_data, desc="Evaluating"):
        # Greedy decoding for evaluation
        if model_type in ["gpt2", "qwen"]:
            inputs = tokenizer(sample['input_text'], return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            input_len = inputs['input_ids'].shape[1]
            response = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        else:
            input_ids = tokenizer.encode(sample['input_text'])
            input_ids = torch.tensor([input_ids], device=device)
            input_len = input_ids.shape[1]
            
            # Get block_size from model config
            block_size = model.config.block_size

            for _ in range(config.max_new_tokens):
                # Safety check: stop if approaching block_size limit
                if input_ids.shape[1] >= block_size - 1:
                    break
                
                output = model(input_ids)
                if isinstance(output, tuple):
                    logits = output[0][:, -1, :]
                else:
                    logits = output[:, -1, :]

                next_token = logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                resp_text = tokenizer.decode(input_ids[0, input_len:].tolist())
                if "proof complete" in resp_text or "no goals" in resp_text:
                    break

            response = tokenizer.decode(input_ids[0, input_len:].tolist())
        
        pred_tactic = extract_first_tactic(response)
        expected = sample['expected_tactic']
        task_type = sample['task_type']
        
        task_total[task_type] += 1
        if pred_tactic == expected:
            task_correct[task_type] += 1
    
    model.train()
    
    # Compute accuracies
    results = {}
    total_correct = sum(task_correct.values())
    total_samples = sum(task_total.values())
    results['overall_accuracy'] = total_correct / max(total_samples, 1) * 100
    
    for task in TASK_TYPES:
        if task_total[task] > 0:
            results[f'acc_{task}'] = task_correct[task] / task_total[task] * 100
        else:
            results[f'acc_{task}'] = 0.0
    
    return results


# ============================================================
# Main training loop
# ============================================================

def train_grpo(
    model,
    tokenizer,
    train_dataset: RLDataset,
    eval_dataset: RLDataset,
    eval_dataset_2: RLDataset,
    config: GRPOConfig,
    output_dir: Path,
    device: str,
    test_samples: int,
    model_type: str = "gpt2",
):
    """Main GRPO training loop."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create reference model (frozen copy)
    # Use deepcopy but move to CPU to save GPU VRAM, only load to GPU when needed
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    # Keep ref_model on same device — for small models this is fine;
    # for large models consider offloading to CPU
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Metrics logging
    metrics_file = output_dir / "grpo_metrics.csv"
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss', 'reward_mean', 'reward_std', 'correct_rate'])
    
    # Training loop
    total_steps = len(train_dataset) * config.epochs // config.batch_size
    step = 0
    
    print(f"\n{'='*60}")
    print(f"GRPO Training")
    print(f"{'='*60}")
    print(f"Model: {model_type}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)} + {len(eval_dataset_2)} (2 sets)")
    print(f"Num generations per prompt: {config.num_generations}")
    print(f"Epochs: {config.epochs}")
    print(f"Total steps: {total_steps}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    best_acc = 0.0
    
    for epoch in range(config.epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")
        
        # Shuffle data
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        
        epoch_losses = []
        epoch_rewards = []
        epoch_correct = 0
        epoch_total = 0
        
        pbar = tqdm(range(0, len(indices), config.batch_size), desc=f"Epoch {epoch+1}")
        skipped_count = 0
        
        for batch_start in pbar:
            batch_indices = indices[batch_start:batch_start + config.batch_size]
            batch_loss = None
            contributing_samples = 0
            batch_rewards = []
            
            for idx in batch_indices:
                sample = train_dataset[idx]
                
                # Check if input is too long (leave room for response)
                if model_type == "custom":
                    input_tokens = tokenizer.encode(sample['input_text'])
                    # Reserve at least 30 tokens for response
                    if len(input_tokens) > model.config.block_size - 30:
                        skipped_count += 1
                        continue
                
                # 1. Generate multiple responses
                model.eval()
                responses, _ = generate_responses(
                    model, tokenizer, sample['input_text'],
                    config.num_generations, config, device, model_type
                )
                model.train()
                
                # 2. Compute rewards and advantages
                rewards, advantages = compute_rewards_batch(
                    responses, sample['expected_tactic'], config
                )
                batch_rewards.extend(rewards)
                epoch_correct += sum(1 for r in rewards if r == config.correct_reward)
                epoch_total += len(rewards)
                
                # 3. Compute policy loss
                loss = compute_policy_loss(
                    model, tokenizer, sample['input_text'],
                    responses, advantages, ref_model,
                    config, device, model_type
                )
                if loss is None:
                    continue
                batch_loss = loss if batch_loss is None else (batch_loss + loss)
                contributing_samples += 1
            
            # Backprop — scale by gradient accumulation steps
            if batch_loss is None or contributing_samples == 0:
                skipped_count += len(batch_indices)
                continue
            batch_loss = batch_loss / contributing_samples / config.gradient_accumulation
            batch_loss.backward()

            if (step + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_losses.append(batch_loss.item())
            epoch_rewards.extend(batch_rewards)
            
            # Update progress bar
            correct_rate = epoch_correct / max(epoch_total, 1) * 100
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'reward': f'{np.mean(batch_rewards):.2f}',
                'correct': f'{correct_rate:.1f}%'
            })
            
            step += 1
            
            # Report skipped samples periodically
            if skipped_count > 0 and step % 100 == 0:
                print(f"\n[Warning] Skipped {skipped_count} samples due to length > {model.config.block_size - 30}")
            
            # Log metrics
            if step % 10 == 0:
                with open(metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        step,
                        f'{batch_loss.item():.6f}',
                        f'{np.mean(batch_rewards):.4f}',
                        f'{np.std(batch_rewards):.4f}',
                        f'{correct_rate:.2f}'
                    ])
            
            # Evaluation (adaptive interval)
            # Early stage (step 0-100): eval every 5 steps
            # Later stage (step > 100): eval every 50 steps
            should_eval = False
            if step <= config.eval_early_threshold:
                should_eval = (step % config.eval_interval_early == 0)
            else:
                should_eval = (step % config.eval_interval_later == 0)
            
            if should_eval:
                print(f"\n[Step {step}] Running evaluation...")
                eval_results = evaluate(
                    model, tokenizer, eval_dataset.samples[:200],
                    config, device, model_type
                )
                eval_results_2 = evaluate(
                    model, tokenizer, eval_dataset_2.samples[:200],
                    config, device, model_type
                )
                
                avg_acc = (eval_results['overall_accuracy'] + eval_results_2['overall_accuracy']) / 2
                print(f"  Set1: {eval_results['overall_accuracy']:.1f}%  Set2: {eval_results_2['overall_accuracy']:.1f}%  Avg: {avg_acc:.1f}%")
                simple_acc = np.mean([eval_results[f'acc_{t}'] for t in SIMPLE_TASKS])
                comp_acc = np.mean([eval_results[f'acc_{t}'] for t in COMPOSITION_TASKS])
                print(f"  Simple: {simple_acc:.1f}%  |  Composition: {comp_acc:.1f}%")
                
                # Save best model (using average of both sets)
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    if model_type in ["gpt2", "qwen"]:
                        model.save_pretrained(output_dir / "best_model")
                        tokenizer.save_pretrained(output_dir / "best_model")
                    else:
                        torch.save(model.state_dict(), output_dir / "best_model.pt")
                    print(f"  New best (avg)! Saved to {output_dir / 'best_model'}")
                
                # Save eval results
                with open(output_dir / f"eval_step_{step}.json", 'w') as f:
                    json.dump({'set1': eval_results, 'set2': eval_results_2, 'avg_accuracy': avg_acc}, f, indent=2)
        
        # Epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg loss: {np.mean(epoch_losses):.4f}")
        print(f"  Avg reward: {np.mean(epoch_rewards):.3f}")
        print(f"  Correct rate: {epoch_correct / max(epoch_total, 1) * 100:.1f}%")
    
    # Final save
    if model_type in ["gpt2", "qwen"]:
        model.save_pretrained(output_dir / "final_model")
        tokenizer.save_pretrained(output_dir / "final_model")
    else:
        torch.save(model.state_dict(), output_dir / "final_model.pt")

    # Final test evaluation on ALL task types (aligned with regular experiments)
    print(f"\n{'='*60}")
    print(f"Final Test Evaluation (ALL task types, {test_samples}/task)")
    print(f"{'='*60}")

    # Test set 1 (seed_base=999_999_000, same as regular experiments)
    test_dataset = RLDataset(TASK_TYPES, test_samples, seed_base=999_999_000)
    test_results = evaluate(
        model, tokenizer, test_dataset.samples,
        config, device, model_type
    )
    test_results['test_samples_per_task'] = test_samples
    test_results['seed_base'] = 999_999_000

    print(f"\nTest Set 1 Results:")
    simple_acc = np.mean([test_results[f'acc_{t}'] for t in SIMPLE_TASKS])
    comp_acc = np.mean([test_results[f'acc_{t}'] for t in COMPOSITION_TASKS])
    print(f"  Overall: {test_results['overall_accuracy']:.1f}%")
    print(f"  Simple avg: {simple_acc:.1f}%  |  Composition avg: {comp_acc:.1f}%")
    for task in TASK_TYPES:
        print(f"    {task}: {test_results[f'acc_{task}']:.1f}%")

    with open(output_dir / "test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)

    # Test set 2 (seed_base=666_666_000, same as regular experiments)
    test_dataset_2 = RLDataset(TASK_TYPES, test_samples, seed_base=666_666_000)
    test_results_2 = evaluate(
        model, tokenizer, test_dataset_2.samples,
        config, device, model_type
    )
    test_results_2['test_samples_per_task'] = test_samples
    test_results_2['seed_base'] = 666_666_000

    print(f"\nTest Set 2 Results:")
    simple_acc_2 = np.mean([test_results_2[f'acc_{t}'] for t in SIMPLE_TASKS])
    comp_acc_2 = np.mean([test_results_2[f'acc_{t}'] for t in COMPOSITION_TASKS])
    print(f"  Overall: {test_results_2['overall_accuracy']:.1f}%")
    print(f"  Simple avg: {simple_acc_2:.1f}%  |  Composition avg: {comp_acc_2:.1f}%")

    with open(output_dir / "test_results_set2.json", 'w') as f:
        json.dump(test_results_2, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best eval accuracy: {best_acc:.1f}%")
    print(f"Results saved to: {output_dir}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Proof Generation")
    
    # Model
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["gpt2", "qwen", "custom"],
                        help="Model type")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to pretrained/finetuned model")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to config.json (for custom models)")
    
    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    
    # Data
    parser.add_argument("--task-mode", type=str, default="all",
                        choices=["all", "simple_only", "composition_only"],
                        help="Which tasks to train on")
    parser.add_argument("--train-samples", type=int, default=500,
                        help="Training samples per task")
    parser.add_argument("--eval-samples", type=int, default=50,
                        help="Eval samples per task")
    parser.add_argument("--test-samples", type=int, default=100,
                        help="Test samples per task (final evaluation)")
    
    # GRPO config
    parser.add_argument("--num-generations", type=int, default=8,
                        help="Number of generations per prompt")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (prompts)")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Build config
    config = GRPOConfig(
        num_generations=args.num_generations,
        temperature=args.temperature,
        learning_rate=args.lr,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    
    # Determine task types
    if args.task_mode == "simple_only":
        task_types = SIMPLE_TASKS
    elif args.task_mode == "composition_only":
        task_types = COMPOSITION_TASKS
    else:
        task_types = TASK_TYPES
    
    # Load model
    print(f"Loading {args.model_type} model from {args.model_path}...")
    if args.model_type == "gpt2":
        model, tokenizer = load_model_gpt2(args.model_path, args.device)
    elif args.model_type == "qwen":
        model, tokenizer = load_model_qwen(args.model_path, args.device)
    else:
        if args.config_path is None:
            args.config_path = str(Path(args.model_path).parent / "config.json")
        model, tokenizer = load_model_custom(args.model_path, args.config_path, args.device)
    
    print(f"Model loaded!")
    
    # Create datasets
    # Training: only the selected task types
    print(f"Creating datasets (task_mode={args.task_mode})...")
    train_dataset = RLDataset(task_types, args.train_samples, seed_base=args.seed * 1000)
    # Eval: ALL task types (to detect forgetting of simple tasks during RL)
    # Two eval sets to match train_circuit_ablation.py
    eval_dataset = RLDataset(TASK_TYPES, args.eval_samples, seed_base=888_888_000)
    eval_dataset_2 = RLDataset(TASK_TYPES, args.eval_samples, seed_base=777_777_000)
    print(f"Train: {len(train_dataset)} samples ({len(task_types)} task types)")
    print(f"Eval: {len(eval_dataset)} + {len(eval_dataset_2)} samples (2 sets, ALL {len(TASK_TYPES)} task types)")
    test_samples = args.test_samples
    
    # Train
    train_grpo(
        model, tokenizer, train_dataset, eval_dataset, eval_dataset_2,
        config, Path(args.output_dir), args.device, args.test_samples, args.model_type
    )


if __name__ == "__main__":
    main()
