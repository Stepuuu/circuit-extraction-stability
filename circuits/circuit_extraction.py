#!/usr/bin/env python3
"""
Phase 1: Circuit Extraction via Learnable Mask Optimization

Following Gao et al. (2025) - "Weight-sparse transformers have interpretable circuits"

For each task, we:
1. Freeze the trained model
2. Insert learnable continuous masks on attention heads and MLP neurons
3. Optimize masks to minimize task loss + L0 penalty (encourages sparsity)
4. Binarize masks to get the final discrete circuit

Node granularity:
  - Attention head: (layer, head) — masks the full (Q,K,V,O) of that head
  - MLP neuron: (layer, neuron_idx) — masks individual neurons in the MLP hidden layer
  
Mean ablation: Nodes "outside" the circuit get replaced by their mean activation
computed over a reference dataset. This is the gold-standard for circuit discovery.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# Insert project root into path (works whether invoked from any directory)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _project_root)

from runtime.inference.gpt import GPT, GPTConfig
from training.train_grpo import (
    RLDataset, TASK_TYPES, SIMPLE_TASKS, COMPOSITION_TASKS,
    generate_sample, get_expected_tactic, extract_first_tactic
)

# ============================================================
# Configuration
# ============================================================

@dataclass
class CircuitExtractionConfig:
    """Configuration for circuit extraction."""
    # Mask optimization
    mask_lr: float = 0.1               # Learning rate for mask parameters
    lambda_l0: float = 0.02            # L0 regularization strength
    num_epochs: int = 150              # Optimization epochs
    temperature: float = 2/3           # Hard-concrete temperature (Louizos et al. 2018)
    stretch_lo: float = -0.1           # Hard-concrete stretch parameters
    stretch_hi: float = 1.1
    
    # Task data
    num_samples_per_task: int = 100    # Samples used for mask optimization
    seed_base: int = 123_456_000       # Seed for generating circuit extraction data
    
    # Circuit thresholding
    threshold: float = 0.5             # Binarization threshold

    # Supervision mode for mask gradient
    # "all"    : CE loss on every output token (old default, biased by format words)
    # "answer" : skip the first output token ("exact"/"apply" — always predictable
    #             from task type, contributes near-zero information gradient);
    #             supervise remaining output tokens (variable names, tactic names)
    # "lastN"  : supervise only the last N output tokens (e.g. "last1", "last2")
    supervision: str = "answer"        # default: focus gradient on answer tokens

    # Mean ablation reference
    num_reference_samples: int = 200   # Samples for computing mean activations
    reference_seed_base: int = 111_111_000
    
    # Hardware
    device: str = "cuda"
    batch_size: int = 16


# ============================================================
# Post-hoc threshold utility
# ============================================================

def rethreshold_circuit(circuit: dict, new_threshold: float,
                        n_head: int, d_head: int) -> dict:
    """Re-apply a different binarization threshold to a saved circuit.

    Mask optimization is NOT re-run — only the final binarization changes.
    The saved `attn_mask_values` / `mlp_mask_values` are raw z-values from
    the Hard-Concrete deterministic forward (range ≈ [−0.1, 1.1]); a node is
    included iff z > new_threshold.

    Accuracy metrics are left as -1.0 (Phase 4 will re-evaluate via mean-
    ablation knockout, which does not require the mask optimizer).
    """
    import copy
    new_circ = copy.deepcopy(circuit)
    new_attn: list = []
    new_mlp:  list = []

    for l_str, z_vals in circuit['attn_mask_values'].items():
        l = int(l_str)
        for h in range(n_head):
            for c in range(d_head):
                if z_vals[h * d_head + c] > new_threshold:
                    new_attn.append([l, h, c])

    for l_str, z_vals in circuit['mlp_mask_values'].items():
        l = int(l_str)
        for n_idx, v in enumerate(z_vals):
            if v > new_threshold:
                new_mlp.append([l, n_idx])

    new_circ['attention_channels']    = new_attn
    new_circ['mlp_neurons']           = new_mlp
    new_circ['num_attention_channels'] = len(new_attn)
    new_circ['num_mlp_neurons']        = len(new_mlp)
    new_circ['total_nodes']            = len(new_attn) + len(new_mlp)
    total_possible = circuit.get('total_possible', 1) or 1
    new_circ['circuit_fraction'] = new_circ['total_nodes'] / total_possible
    for key in ('circuit_accuracy', 'circuit_exact_match',
                'accuracy_retention', 'exact_match_retention', 'circuit_edge_fraction'):
        new_circ[key] = -1.0
    return new_circ


# ============================================================
# Hard Concrete Distribution (Louizos et al., 2018)
# ============================================================

class HardConcreteMask(nn.Module):
    """
    Learnable mask using the Hard Concrete distribution.
    
    This gives us a differentiable approximation to a Bernoulli mask,
    with support on [0, 1] and mass at exactly 0 and exactly 1.
    
    The L0 norm (expected number of non-zero entries) is differentiable.
    """
    def __init__(self, n_masks: int, temperature: float = 2/3,
                 stretch_lo: float = -0.1, stretch_hi: float = 1.1,
                 init_value: float = 5.0):
        super().__init__()
        # Initialize log-alpha so that masks start mostly "on"
        self.log_alpha = nn.Parameter(torch.full((n_masks,), init_value))
        self.temperature = temperature
        self.stretch_lo = stretch_lo
        self.stretch_hi = stretch_hi
    
    def forward(self, deterministic: bool = False) -> torch.Tensor:
        """
        Returns mask values in [0, 1].
        When deterministic=True (eval), uses the expected mask.
        """
        if deterministic or not self.training:
            # Sigmoid gives P(mask > 0)
            z = torch.sigmoid(self.log_alpha)
            z = z * (self.stretch_hi - self.stretch_lo) + self.stretch_lo
            return z.clamp(0.0, 1.0)
        
        # Sample from Hard Concrete
        u = torch.rand_like(self.log_alpha).clamp(1e-8, 1 - 1e-8)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.temperature)
        s = s * (self.stretch_hi - self.stretch_lo) + self.stretch_lo
        z = s.clamp(0.0, 1.0)
        return z
    
    def l0_loss(self) -> torch.Tensor:
        """Expected L0 norm (number of non-zero masks)."""
        # P(z > 0) = sigmoid(log_alpha - temperature * log(-stretch_lo / stretch_hi))
        return torch.sigmoid(
            self.log_alpha - self.temperature * torch.log(
                torch.tensor(-self.stretch_lo / self.stretch_hi)
            )
        ).sum()
    
    def get_binary_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """Get binarized mask."""
        z = self.forward(deterministic=True)
        return (z > threshold).float()
    
    def sparsity(self) -> float:
        """Fraction of masks that are off."""
        binary = self.get_binary_mask()
        return 1.0 - binary.mean().item()


# ============================================================
# Masked Model Wrapper
# ============================================================

class MaskedModel(nn.Module):
    """
    Wraps a frozen GPT model with learnable masks on:
      - Attention heads: per (layer, head) — binary on/off for each head
      - MLP neurons: per (layer, neuron) — binary on/off for each hidden neuron
    
    Masked-out components are replaced with their mean activations (mean ablation).
    """
    
    def __init__(self, model: GPT, config: CircuitExtractionConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.n_layer = model.config.n_layer
        self.n_head = model.config.n_head
        self.d_head = model.config.d_head
        self.d_mlp = model.config.d_mlp
        self.d_model = model.config.d_model
        
        # Freeze original model
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Learnable masks
        # Attention channel masks: [n_layer * n_head * d_head]  ← Gao-style
        # Each attention channel is one d_head dimension; d_head=16 channels per head
        # This matches Gao et al.'s definition: node = individual attention channel (row/col of weight matrix)
        self.attn_masks = HardConcreteMask(
            self.n_layer * self.n_head * self.d_head,
            temperature=config.temperature,
            stretch_lo=config.stretch_lo,
            stretch_hi=config.stretch_hi,
        )
        
        # MLP neuron masks: [n_layer * d_mlp]
        self.mlp_masks = HardConcreteMask(
            self.n_layer * self.d_mlp,
            temperature=config.temperature,
            stretch_lo=config.stretch_lo,
            stretch_hi=config.stretch_hi,
        )
        
        # Mean activations for ablation (computed separately)
        # attn_means[l]: shape (n_head, d_head) — mean output per head
        self.attn_means: Dict[int, torch.Tensor] = {}
        # mlp_means[l]: shape (d_mlp,) — mean post-activation per neuron
        self.mlp_means: Dict[int, torch.Tensor] = {}
    
    def set_mean_activations(self, attn_means: Dict[int, torch.Tensor],
                              mlp_means: Dict[int, torch.Tensor]):
        """Set precomputed mean activations for ablation."""
        self.attn_means = attn_means
        self.mlp_means = mlp_means
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass with masked attention heads and MLP neurons.
        """
        device = idx.device
        b, t = idx.size()
        
        # Embedding
        tok_emb = self.model.transformer.wte(idx)
        pos_emb = self.model.transformer.wpe.weight[:t].unsqueeze(0)
        x = self.model.transformer.drop(tok_emb + pos_emb)
        
        # Get mask values
        attn_mask_vals = self.attn_masks()  # [n_layer * n_head]
        mlp_mask_vals = self.mlp_masks()    # [n_layer * d_mlp]
        
        # Process each layer
        for layer_idx, block in enumerate(self.model.transformer.h):
            # ---- Attention with head masking ----
            x = self._masked_attn_block(x, block, layer_idx, attn_mask_vals)
            
            # ---- MLP with neuron masking ----
            x = self._masked_mlp_block(x, block, layer_idx, mlp_mask_vals)
        
        # Final layer norm + LM head
        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x) + self.model.final_logits_bias
        
        if self.model.config.enable_bigram_table:
            additional = F.embedding(idx, self.model.bigram_table, padding_idx=-1)
            logits = logits + additional.to(x.dtype)
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1
            )
        else:
            loss = torch.zeros(1, device=device)
        
        return logits, loss
    
    def _masked_attn_block(self, x, block, layer_idx, attn_mask_vals):
        """Attention block with per-head masking."""
        B, T, C = x.size()
        
        # Pre-norm
        x_norm = block.ln_1(x)
        
        # Compute Q, K, V
        qkv = block.attn.c_attn(x_norm)
        q, k, v = qkv.split(self.n_head * self.d_head, dim=2)
        
        # Reshape to (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        # Standard attention
        scale = 1.0 / (self.d_head ** 0.5)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )  # (B, n_head, T, d_head)
        
        # Apply per-channel masks (Gao-style: individual attention channel = one d_head dimension)
        # Slice n_head * d_head channels for this layer, reshape to (1, n_head, 1, d_head)
        ch_start = layer_idx * self.n_head * self.d_head
        ch_end   = ch_start + self.n_head * self.d_head
        channel_masks = attn_mask_vals[ch_start:ch_end]
        channel_masks = channel_masks.view(1, self.n_head, 1, self.d_head)  # broadcast over (B, T)

        # Mean ablation for masked channels
        if layer_idx in self.attn_means:
            mean_val = self.attn_means[layer_idx].to(y.device)  # (n_head, d_head)
            mean_val = mean_val.view(1, self.n_head, 1, self.d_head).expand_as(y)
            y = channel_masks * y + (1 - channel_masks) * mean_val
        else:
            y = channel_masks * y
        
        # Reassemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.d_head)
        y = block.attn.c_proj(y)
        y = block.attn.resid_dropout(y)
        
        # Residual connection
        x = x + y
        return x
    
    def _masked_mlp_block(self, x, block, layer_idx, mlp_mask_vals):
        """MLP block with per-neuron masking."""
        B, T, C = x.size()
        
        # Pre-norm
        x_norm = block.ln_2(x)
        
        # MLP forward: c_fc -> activation -> mask -> c_proj
        h = block.mlp.c_fc(x_norm)
        h = block.mlp.act_fn(h)  # (B, T, d_mlp)
        
        # Apply per-neuron masks
        neuron_masks = mlp_mask_vals[layer_idx * self.d_mlp : (layer_idx + 1) * self.d_mlp]
        neuron_masks = neuron_masks.view(1, 1, self.d_mlp)  # broadcast
        
        # Mean ablation for masked neurons
        if layer_idx in self.mlp_means:
            mean_val = self.mlp_means[layer_idx].to(h.device)  # (d_mlp,)
            mean_val = mean_val.view(1, 1, self.d_mlp).expand_as(h)
            h = neuron_masks * h + (1 - neuron_masks) * mean_val
        else:
            h = neuron_masks * h
        
        # Project back and residual
        y = block.mlp.c_proj(h)
        y = block.mlp.dropout(y)
        
        x = x + y
        return x
    
    def l0_loss(self) -> torch.Tensor:
        """Total L0 across all masks."""
        return self.attn_masks.l0_loss() + self.mlp_masks.l0_loss()
    
    def total_nodes(self) -> int:
        """Total maskable nodes (Gao definition).
        attn channels: n_layer * n_head * d_head
        MLP neurons  : n_layer * d_mlp
        """
        return self.n_layer * self.n_head * self.d_head + self.n_layer * self.d_mlp
    
    def get_circuit(self, threshold: float = 0.5) -> dict:
        """Extract the binary circuit using Gao et al.'s node definition.

        Node types (Gao 2025):
          - attention channel: one dimension of d_head inside a (layer, head)
            → (layer, head, channel) tuple, total = n_layer * n_head * d_head
          - MLP neuron: one hidden unit in MLP
            → (layer, neuron) tuple, total = n_layer * d_mlp
        Edge = nonzero scalar weight entry (approximated by L0frac × node presence).
        """
        attn_binary = self.attn_masks.get_binary_mask(threshold)  # (n_layer*n_head*d_head,)
        mlp_binary  = self.mlp_masks.get_binary_mask(threshold)   # (n_layer*d_mlp,)

        circuit = {
            'attention_channels': [],  # List of (layer, head, channel) tuples  ← Gao node
            'mlp_neurons': [],          # List of (layer, neuron) tuples
            'attn_mask_values': {},     # Raw mask values per layer
            'mlp_mask_values': {},      # Raw mask values per layer
        }

        attn_vals_all = self.attn_masks.forward(deterministic=True)
        mlp_vals_all  = self.mlp_masks.forward(deterministic=True)

        for l in range(self.n_layer):
            # Attention channels for this layer: n_head * d_head values
            a_start = l * self.n_head * self.d_head
            layer_attn = attn_binary[a_start : a_start + self.n_head * self.d_head]
            attn_vals  = attn_vals_all[a_start : a_start + self.n_head * self.d_head]
            circuit['attn_mask_values'][l] = attn_vals.detach().cpu().tolist()

            for h in range(self.n_head):
                for c in range(self.d_head):
                    if layer_attn[h * self.d_head + c] > 0.5:
                        circuit['attention_channels'].append((l, h, c))

            # MLP neurons for this layer
            m_start = l * self.d_mlp
            layer_mlp = mlp_binary[m_start : m_start + self.d_mlp]
            mlp_vals  = mlp_vals_all[m_start : m_start + self.d_mlp]
            circuit['mlp_mask_values'][l] = mlp_vals.detach().cpu().tolist()

            for n in range(self.d_mlp):
                if layer_mlp[n] > 0.5:
                    circuit['mlp_neurons'].append((l, n))

        circuit['num_attention_channels'] = len(circuit['attention_channels'])
        circuit['num_mlp_neurons']        = len(circuit['mlp_neurons'])
        circuit['total_nodes']    = circuit['num_attention_channels'] + circuit['num_mlp_neurons']
        circuit['total_possible'] = self.total_nodes()
        circuit['circuit_fraction'] = circuit['total_nodes'] / circuit['total_possible']

        return circuit


# ============================================================
# Data Preparation
# ============================================================

class TokenizerWrapper:
    """Wrapper for SentencePiece tokenizer."""
    def __init__(self, sp_model):
        self.sp_model = sp_model
    def encode(self, text):
        return self.sp_model.encode(text, out_type=int)
    def decode(self, ids):
        return self.sp_model.decode(ids)
    def __len__(self):
        return self.sp_model.vocab_size()


# Fixed epilogue token IDs: "\nstate_1:\nno goals\nproof complete"
# These are invariant across all samples and carry zero reasoning signal.
_EPILOGUE_TOKEN_IDS = None  # lazily populated

def _get_epilogue_token_ids(tokenizer) -> List[int]:
    """Return token IDs for the fixed epilogue suffix.
    
    Uses differential encoding to avoid SentencePiece BOS artifacts:
    encode("X" + epilogue) minus encode("X").
    """
    global _EPILOGUE_TOKEN_IDS
    if _EPILOGUE_TOKEN_IDS is None:
        epilogue_text = "\nstate_1:\nno goals\nproof complete"
        prefix = "X"
        full_ids = tokenizer.encode(prefix + epilogue_text)
        prefix_ids = tokenizer.encode(prefix)
        # The epilogue token IDs are the suffix after the prefix
        _EPILOGUE_TOKEN_IDS = full_ids[len(prefix_ids):]
    return _EPILOGUE_TOKEN_IDS


def _find_epilogue_start(tokens: List[int], epilogue_ids: List[int]) -> int:
    """Find where the epilogue starts in the token list. Returns index or len(tokens)."""
    elen = len(epilogue_ids)
    for i in range(len(tokens) - elen, -1, -1):  # search from end
        if tokens[i:i + elen] == epilogue_ids:
            return i
    return len(tokens)  # not found → don't trim


def _build_target_ids(
    tokens: List[int],
    input_len: int,
    block_size: int,
    supervision: str,
    epilogue_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """Build target_ids tensor for CE loss, respecting supervision mode.

    supervision modes:
      "all"    — supervise every output token position (old default)
      "answer" — skip the FIRST output token (always a format word: "exact" / "apply");
                 supervise all remaining output tokens including epilogue.
      "tactic" — skip the first format word AND the fixed epilogue suffix
                 ("state_1:\nno goals\nproof complete"). Only supervises
                 reasoning-relevant tokens: hypothesis names, tactic names,
                 intermediate structure. This is the recommended mode.
      "lastN"  — supervise only the last N output positions (e.g. "last1", "last2").
    """
    target_ids = torch.full((block_size,), -1, dtype=torch.long)
    # output positions in next-token-prediction space: [input_len-1, len(tokens)-2]
    out_start = input_len - 1     # position that predicts first output token
    out_end   = min(len(tokens) - 1, block_size)  # last supervised position (exclusive)

    if supervision == "all":
        sup_start = out_start
        sup_end = out_end
    elif supervision == "answer":
        # Skip position out_start (predicts "exact"/"apply" — format word)
        sup_start = out_start + 1
        sup_end = out_end
    elif supervision == "tactic":
        # Skip first format word AND fixed epilogue at the end
        sup_start = out_start + 1
        if epilogue_ids is not None:
            epi_start = _find_epilogue_start(tokens, epilogue_ids)
            # In next-token-prediction space, position (epi_start - 1) predicts
            # the first epilogue token. We stop supervision before that.
            sup_end = min(out_end, epi_start - 1)
        else:
            sup_end = out_end
    elif supervision.startswith("last"):
        try:
            n = int(supervision[4:])
        except ValueError:
            n = 1
        sup_start = max(out_start, out_end - n)
        sup_end = out_end
    else:
        sup_start = out_start  # fallback
        sup_end = out_end

    for i in range(sup_start, sup_end):
        target_ids[i] = tokens[i + 1]

    return target_ids


def prepare_task_data(
    task_types: List[str],
    num_samples: int,
    seed_base: int,
    tokenizer,
    block_size: int,
    device: str,
    supervision: str = "answer",
) -> Dict[str, List[Dict]]:
    """
    Generate and tokenize task-specific data.

    Returns dict: task_type -> list of {input_ids, target_ids, input_text, output_text}

    supervision controls which output tokens contribute to the CE loss gradient:
      "all"    — every output token (legacy; biased by uninformative format words)
      "answer" — skip first output token ("exact"/"apply"); focus on variable names
      "tactic" — skip first format word AND fixed epilogue; recommended mode
      "lastN"  — last N output tokens only
    """
    data = {}
    
    # Pre-compute epilogue token IDs for 'tactic' supervision mode
    epilogue_ids = _get_epilogue_token_ids(tokenizer) if supervision == "tactic" else None
    
    for task_type in task_types:
        samples = []
        seed = seed_base
        collected = 0
        attempts = 0
        
        while collected < num_samples and attempts < num_samples * 10:
            try:
                input_text, output_text = generate_sample(task_type, seed)
                expected_tactic = get_expected_tactic(output_text)
                
                # Tokenize full sequence: input + output
                full_text = input_text + output_text
                tokens = tokenizer.encode(full_text)
                
                if len(tokens) > block_size:
                    seed += 1
                    attempts += 1
                    continue
                
                input_tokens = tokenizer.encode(input_text)
                target_start = len(input_tokens)

                padded = tokens + [0] * (block_size - len(tokens))
                input_ids = torch.tensor(padded[:block_size], dtype=torch.long)

                target_ids = _build_target_ids(tokens, target_start, block_size, supervision, epilogue_ids)
                
                samples.append({
                    'input_ids': input_ids,
                    'target_ids': target_ids,
                    'input_text': input_text,
                    'output_text': output_text,
                    'expected_tactic': expected_tactic,
                    'task_type': task_type,
                })
                collected += 1
            except Exception:
                pass
            
            seed += 1
            attempts += 1
        
        data[task_type] = samples
        print(f"  {task_type}: {len(samples)} samples  [supervision={supervision}]")
    
    return data


# ============================================================
# Mean Activation Computation
# ============================================================

@torch.no_grad()
def compute_mean_activations(
    model: GPT,
    tokenizer,
    num_samples: int,
    seed_base: int,
    block_size: int,
    device: str,
    batch_size: int = 16,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Compute mean activations for each attention head output and MLP neuron,
    over a reference dataset. Used for mean ablation.
    
    Returns:
        attn_means: {layer_idx: tensor of shape (n_head, d_head)}
        mlp_means: {layer_idx: tensor of shape (d_mlp,)}
    """
    model.eval()
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    d_head = model.config.d_head
    d_mlp = model.config.d_mlp
    
    # Accumulators
    attn_sums = {l: torch.zeros(n_head, d_head, device=device) for l in range(n_layer)}
    mlp_sums = {l: torch.zeros(d_mlp, device=device) for l in range(n_layer)}
    count = 0
    
    # Generate reference data from all tasks
    print("Computing mean activations for mean ablation...")
    all_inputs = []
    seed = seed_base
    for _ in range(num_samples):
        for task_type in TASK_TYPES:
            try:
                input_text, output_text = generate_sample(task_type, seed)
                full_text = input_text + output_text
                tokens = tokenizer.encode(full_text)
                if len(tokens) <= block_size:
                    padded = tokens + [0] * (block_size - len(tokens))
                    all_inputs.append(torch.tensor(padded[:block_size], dtype=torch.long))
            except Exception:
                pass
            seed += 1
    
    # Process in batches
    for batch_start in range(0, len(all_inputs), batch_size):
        batch = torch.stack(all_inputs[batch_start:batch_start + batch_size]).to(device)
        B, T = batch.size()
        
        # Forward through embeddings
        tok_emb = model.transformer.wte(batch)
        pos_emb = model.transformer.wpe.weight[:T].unsqueeze(0)
        x = model.transformer.drop(tok_emb + pos_emb)
        
        # Through each layer, collect activations
        for l, block in enumerate(model.transformer.h):
            # Attention
            x_norm = block.ln_1(x)
            qkv = block.attn.c_attn(x_norm)
            q, k, v = qkv.split(n_head * d_head, dim=2)
            
            q = q.view(B, T, n_head, d_head).transpose(1, 2)
            k = k.view(B, T, n_head, d_head).transpose(1, 2)
            v = v.view(B, T, n_head, d_head).transpose(1, 2)
            
            scale = 1.0 / (d_head ** 0.5)
            y_attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=scale)
            # y_attn: (B, n_head, T, d_head)
            
            # Accumulate mean over (B, T) dimensions
            attn_sums[l] += y_attn.mean(dim=(0, 2))  # (n_head, d_head)
            
            # Complete attention residual
            y_assembled = y_attn.transpose(1, 2).contiguous().view(B, T, n_head * d_head)
            y_proj = block.attn.c_proj(y_assembled)
            x = x + y_proj
            
            # MLP
            x_norm2 = block.ln_2(x)
            h = block.mlp.c_fc(x_norm2)
            h = block.mlp.act_fn(h)  # (B, T, d_mlp)
            
            # Accumulate mean over (B, T)
            mlp_sums[l] += h.mean(dim=(0, 1))  # (d_mlp,)
            
            # Complete MLP residual
            y_mlp = block.mlp.c_proj(h)
            x = x + y_mlp
        
        count += 1
    
    # Average
    attn_means = {l: attn_sums[l] / count for l in range(n_layer)}
    mlp_means = {l: mlp_sums[l] / count for l in range(n_layer)}
    
    print(f"  Mean activations computed from {len(all_inputs)} samples, {count} batches")
    return attn_means, mlp_means


# ============================================================
# Circuit Extraction
# ============================================================

def extract_circuit_for_task(
    model: GPT,
    task_type: str,
    task_data: List[Dict],
    attn_means: Dict[int, torch.Tensor],
    mlp_means: Dict[int, torch.Tensor],
    config: CircuitExtractionConfig,
) -> dict:
    """
    Extract the minimal circuit for a single task via mask optimization.
    """
    device = config.device
    
    print(f"\n{'='*60}")
    print(f"Extracting circuit for: {task_type}")
    print(f"{'='*60}")
    
    # Create masked model
    masked_model = MaskedModel(model, config).to(device)
    masked_model.set_mean_activations(attn_means, mlp_means)
    
    # Optimizer for mask parameters only
    mask_params = list(masked_model.attn_masks.parameters()) + list(masked_model.mlp_masks.parameters())
    optimizer = torch.optim.Adam(mask_params, lr=config.mask_lr)
    
    # Prepare batches
    input_ids = torch.stack([s['input_ids'] for s in task_data]).to(device)
    target_ids = torch.stack([s['target_ids'] for s in task_data]).to(device)
    
    n_samples = len(task_data)
    total_nodes = masked_model.total_nodes()
    
    # ── Evaluate FULL model (no masking) BEFORE optimization ──────────────
    # This lets us confirm: (1) we're using the right model, (2) baseline is high
    print(f"  Evaluating full model baseline (no masking)...")
    with torch.no_grad():
        model.eval()
        full_correct = 0
        full_total = 0
        for batch_start in range(0, n_samples, config.batch_size):
            b_in  = input_ids[batch_start:batch_start + config.batch_size]
            b_tgt = target_ids[batch_start:batch_start + config.batch_size]
            out = model(b_in)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(dim=-1)  # (B, T)
            valid = (b_tgt != -1)
            full_correct += (preds[valid] == b_tgt[valid]).sum().item()
            full_total += valid.sum().item()
    full_model_accuracy = full_correct / max(full_total, 1)
    # Also compute exact match (entire output sequence correct) on full model
    full_exact_correct = 0
    with torch.no_grad():
        model.eval()
        for batch_start in range(0, n_samples, config.batch_size):
            b_in  = input_ids[batch_start:batch_start + config.batch_size]
            b_tgt = target_ids[batch_start:batch_start + config.batch_size]
            out = model(b_in)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(dim=-1)
            for i in range(b_tgt.shape[0]):
                sample_mask = (b_tgt[i] != -1)
                if sample_mask.sum() > 0:
                    if (preds[i][sample_mask] == b_tgt[i][sample_mask]).all():
                        full_exact_correct += 1
    full_model_exact_match = full_exact_correct / max(n_samples, 1)
    print(f"  ✓ Full model accuracy (before masking): per_token={full_model_accuracy:.3f}  exact_match={full_model_exact_match:.3f}")
    if full_model_accuracy < 0.70:
        print(f"  ⚠ WARNING: Full model accuracy is low ({full_model_accuracy:.3f}). "
              "Check checkpoint / tokenizer / data format!")
    # ──────────────────────────────────────────────────────────────────────

    best_em   = -1.0   # primary selection criterion: exact match on circuit
    best_loss = float('inf')  # tiebreaker: CE loss
    best_state = None

    for epoch in range(config.num_epochs):
        masked_model.train()
        total_task_loss = 0
        total_l0 = 0

        # ── Gradient step (CE loss — still needed for differentiable opt) ──
        indices = torch.randperm(n_samples)
        for batch_start in range(0, n_samples, config.batch_size):
            batch_idx    = indices[batch_start:batch_start + config.batch_size]
            batch_input  = input_ids[batch_idx]
            batch_target = target_ids[batch_idx]

            logits, task_loss = masked_model(batch_input, batch_target)
            l0 = masked_model.l0_loss() / total_nodes  # normalized

            loss = task_loss + config.lambda_l0 * l0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_task_loss += task_loss.item()
            total_l0 += l0.item()

        n_batches = max(1, n_samples // config.batch_size)
        avg_task_loss = total_task_loss / n_batches
        avg_l0        = total_l0 / n_batches

        masked_model.eval()
        circuit = masked_model.get_circuit(config.threshold)

        # ── Evaluate EM on current circuit (every 5 epochs) ──────────────
        # Exact match = entire output sequence correct.
        # This is the meaningful criterion: a circuit that gets ONE key token
        # wrong is useless for logical reasoning even if per-token acc is 99%.
        # We still optimize with CE (differentiable); EM guides state saving.
        epoch_em = 0.0
        if epoch % 5 == 0 or epoch == config.num_epochs - 1:
            em_correct = 0
            with torch.no_grad():
                for bs in range(0, n_samples, config.batch_size):
                    b_in  = input_ids[bs:bs + config.batch_size]
                    b_tgt = target_ids[bs:bs + config.batch_size]
                    logits_c, _ = masked_model(b_in, b_tgt)
                    preds_c = logits_c.argmax(dim=-1)
                    for i in range(b_tgt.shape[0]):
                        sm = (b_tgt[i] != -1)
                        if sm.sum() > 0 and (preds_c[i][sm] == b_tgt[i][sm]).all():
                            em_correct += 1
            epoch_em = em_correct / max(n_samples, 1)

        if epoch % 10 == 0 or epoch == config.num_epochs - 1:
            print(f"  Epoch {epoch:3d}: ce={avg_task_loss:.4f}  "
                  f"l0={avg_l0:.4f}  em={epoch_em:.3f}  "
                  f"attn_ch={circuit['num_attention_channels']}/{model.config.n_layer * model.config.n_head * model.config.d_head}  "
                  f"mlp={circuit['num_mlp_neurons']}/{model.config.n_layer * model.config.d_mlp}  "
                  f"frac={circuit['circuit_fraction']:.3f}")

        # ── Save best state based on EM (tiebreak: CE loss) ──────────────
        if epoch % 5 == 0 or epoch == config.num_epochs - 1:
            combined = avg_task_loss + config.lambda_l0 * avg_l0
            if epoch_em > best_em or (epoch_em == best_em and combined < best_loss):
                best_em   = epoch_em
                best_loss = combined
                best_state = {
                    'attn_masks': masked_model.attn_masks.state_dict(),
                    'mlp_masks':  masked_model.mlp_masks.state_dict(),
                }

    # Restore best
    if best_state:
        masked_model.attn_masks.load_state_dict(best_state['attn_masks'])
        masked_model.mlp_masks.load_state_dict(best_state['mlp_masks'])
    
    # Get final circuit
    masked_model.eval()
    final_circuit = masked_model.get_circuit(config.threshold)
    
    # Evaluate circuit quality: run masked model on task data
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_start in range(0, n_samples, config.batch_size):
            batch_input = input_ids[batch_start:batch_start + config.batch_size]
            batch_target = target_ids[batch_start:batch_start + config.batch_size]
            logits, _ = masked_model(batch_input, batch_target)
            
            # Check per-token accuracy on output region
            preds = logits.argmax(dim=-1)  # (B, T)
            mask = (batch_target != -1)
            correct += (preds[mask] == batch_target[mask]).sum().item()
            total += mask.sum().item()
        
        circuit_accuracy = correct / max(total, 1)
    
    # Exact match for circuit
    circuit_exact_correct = 0
    with torch.no_grad():
        masked_model.eval()
        for batch_start in range(0, n_samples, config.batch_size):
            batch_input  = input_ids[batch_start:batch_start + config.batch_size]
            batch_target = target_ids[batch_start:batch_start + config.batch_size]
            logits, _ = masked_model(batch_input, batch_target)
            preds = logits.argmax(dim=-1)
            for i in range(batch_target.shape[0]):
                sample_mask = (batch_target[i] != -1)
                if sample_mask.sum() > 0:
                    if (preds[i][sample_mask] == batch_target[i][sample_mask]).all():
                        circuit_exact_correct += 1
    circuit_exact_match = circuit_exact_correct / max(n_samples, 1)
    
    final_circuit['task_type'] = task_type
    final_circuit['full_model_accuracy'] = full_model_accuracy        # per-token, before masking
    final_circuit['full_model_exact_match'] = full_model_exact_match  # exact match, before masking
    final_circuit['circuit_accuracy'] = circuit_accuracy              # per-token, after masking
    final_circuit['circuit_exact_match'] = circuit_exact_match        # exact match, after masking
    final_circuit['accuracy_retention'] = (
        circuit_accuracy / full_model_accuracy if full_model_accuracy > 0.01 else 0.0
    )
    final_circuit['exact_match_retention'] = (
        circuit_exact_match / full_model_exact_match if full_model_exact_match > 0.01 else 0.0
    )
    final_circuit['config'] = asdict(config)
    
    print(f"\n  Final circuit for {task_type}:")
    print(f"    Attn channels   : {final_circuit['num_attention_channels']}/{model.config.n_layer * model.config.n_head * model.config.d_head}  (Gao node = 1 channel)")
    print(f"    MLP neurons     : {final_circuit['num_mlp_neurons']}/{model.config.n_layer * model.config.d_mlp}")
    print(f"    Total nodes     : {final_circuit['total_nodes']}/{final_circuit['total_possible']}")
    print(f"    Circuit fraction: {final_circuit['circuit_fraction']:.3f}")
    print(f"    Full model acc  : per_token={full_model_accuracy:.3f}  exact_match={full_model_exact_match:.3f}")
    print(f"    Circuit acc     : per_token={circuit_accuracy:.3f}  exact_match={circuit_exact_match:.3f}")
    em_ret = final_circuit['exact_match_retention']
    print(f"    Acc retention   : per_token={final_circuit['accuracy_retention']:.3f}  "
          f"exact_match={em_ret:.3f}  "
          f"({'✓' if em_ret > 0.9 else '⚠ degraded'})")
    
    # ── Edge-weighted circuit size (Gao et al. style) ──────────────────────
    # Training weight sparsity (L0frac) tells us roughly what fraction of scalar
    # weights are non-zero in this model. A node in a sparse model carries far
    # fewer actual computations (edges) than a node in a dense model.
    #
    # node_fraction = selected_nodes / total_nodes  ← our current metric
    #                                                  (misleading across sparsity)
    # edge_fraction ≈ node_fraction × L0frac        ← Gao's metric
    #                                                  (comparable across models)
    #
    # Example: sp90 node_fraction=0.62, L0frac=0.10 → edge_fraction≈0.062
    #          dense  node_fraction=0.28, L0frac=1.00 → edge_fraction≈0.285
    # → sp90 circuit is actually 4.6× lighter than dense in Gao's sense!
    l0frac = getattr(model, 'weight_l0frac', 1.0)
    edge_fraction = final_circuit['circuit_fraction'] * l0frac
    final_circuit['model_weight_l0frac'] = l0frac
    final_circuit['circuit_edge_fraction'] = edge_fraction
    print(f"    Edge-weighted   : node_frac={final_circuit['circuit_fraction']:.3f}  "
          f"× L0frac({l0frac:.2f}) = edge_frac={edge_fraction:.4f}  "
          f"(Gao-style metric, comparable across sparsity levels)")
    # ────────────────────────────────────────────────────────────────────────
    
    return final_circuit


# ============================================================
# Model Loading
# ============================================================

def load_model(checkpoint_path: str, config_path: str, device: str) -> GPT:
    """Load model from checkpoint + config."""
    print(f"Loading model from {checkpoint_path}")
    
    with open(config_path, 'r') as f:
        exp_config = json.load(f)
    
    # Build GPTConfig from experiment config
    ablation_config = exp_config.get('ablation_config', {})
    
    gpt_config = GPTConfig(
        n_layer=exp_config['n_layer'],
        n_head=exp_config['n_head'],
        d_model=exp_config['d_model'],
        d_head=exp_config.get('d_head', exp_config['d_model'] // exp_config['n_head']),
        d_mlp=exp_config.get('d_mlp', 4 * exp_config['d_model']),
        block_size=exp_config['n_ctx'],
        vocab_size=exp_config['vocab_size'],
        bias=True,
        dropout=0.0,
        flash=True,
        rms_norm=str(ablation_config.get('rms_norm', 'False')).lower() in ('true', '1'),
        tied_unembed=str(ablation_config.get('tied_unembed', 'False')).lower() in ('true', '1'),
        enable_bigram_table=True,  # circuit_clean models have bigram table
        learnable_bigram_table=False,
        sink=str(ablation_config.get('sink', 'False')).lower() in ('true', '1'),
        grad_checkpointing=False,
    )
    
    model = GPT(gpt_config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle potential key mismatches
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"  Model: {gpt_config.n_layer}L, {gpt_config.n_head}H, {gpt_config.d_model}D, "
          f"block_size={gpt_config.block_size}")
    
    # Read training sparsity level for edge-weighted circuit size computation
    # L0frac = fraction of scalar weights expected to be non-zero during training
    # dense=1.0, sp25=0.75, sp50=0.50, sp75=0.25, sp90=0.10
    model.weight_l0frac = float(exp_config.get('L0frac', 1.0))
    print(f"  Training L0frac (weight density): {model.weight_l0frac:.3f}  "
          f"(≈ {model.weight_l0frac*100:.0f}% of scalar weights are non-zero)")
    return model


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Circuit Extraction via Mask Optimization")
    
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (best_model.pt)")
    parser.add_argument("--config", required=True, help="Path to model config.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for circuits")
    
    # Task selection
    parser.add_argument("--tasks", nargs="+", default=None,
                       help="Task types to extract circuits for (default: all)")
    parser.add_argument("--task-group", choices=["simple", "composition", "all"], default="all",
                       help="Task group shortcut")
    
    # Hyperparameters
    parser.add_argument("--lambda-l0", type=float, default=0.02, help="L0 penalty")
    parser.add_argument("--num-epochs", type=int, default=150, help="Optimization epochs")
    parser.add_argument("--mask-lr", type=float, default=0.1, help="Mask learning rate")
    parser.add_argument("--num-samples", type=int, default=100, help="Samples per task")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--supervision", type=str, default="tactic",
                       help="Which output tokens drive the CE gradient: "
                            "\"all\" (every output token), "
                            "\"answer\" (skip first output format word; DEFAULT), "
                            "\"lastN\" e.g. last1/last2 (only last N tokens). "
                            "EM is always used for best-state selection regardless.")
    parser.add_argument("--extra-thresholds", nargs="*", type=float, default=[],
                       help="Additional binarization thresholds applied post-hoc from the "
                            "already-optimized mask values (no re-training). Each T produces "
                            "a sibling directory circuits_thr{T}/ next to output-dir. "
                            "Example: --extra-thresholds 0.3 0.7")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    
    # Hardware
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    
    args = parser.parse_args()
    
    if args.device == "cuda":
        args.device = f"cuda:{args.gpu_id}"
    
    # Determine tasks
    if args.tasks:
        tasks = args.tasks
    elif args.task_group == "simple":
        tasks = SIMPLE_TASKS
    elif args.task_group == "composition":
        tasks = COMPOSITION_TASKS
    else:
        tasks = TASK_TYPES
    
    # Config
    config = CircuitExtractionConfig(
        mask_lr=args.mask_lr,
        lambda_l0=args.lambda_l0,
        num_epochs=args.num_epochs,
        num_samples_per_task=args.num_samples,
        threshold=args.threshold,
        supervision=args.supervision,
        device=args.device,
        batch_size=args.batch_size,
    )
    
    # Load model
    model = load_model(args.checkpoint, args.config, args.device)
    block_size = model.config.block_size
    
    # Load tokenizer
    tokenizer_path = "artifacts/tokenizer/trained/tokenizer.model"
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    tokenizer = TokenizerWrapper(sp)
    
    # Prepare task data
    print(f"\nPreparing task data ({config.num_samples_per_task} samples per task, supervision={config.supervision})...")
    task_data = prepare_task_data(
        tasks, config.num_samples_per_task, config.seed_base,
        tokenizer, block_size, args.device, supervision=config.supervision
    )
    
    # Compute mean activations
    attn_means, mlp_means = compute_mean_activations(
        model, tokenizer, config.num_reference_samples,
        config.reference_seed_base, block_size, args.device,
        batch_size=config.batch_size
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract circuits for each task
    all_circuits = {}
    for task_type in tasks:
        if task_type not in task_data or len(task_data[task_type]) < 10:
            print(f"\nSkipping {task_type}: insufficient data")
            continue
        
        circuit = extract_circuit_for_task(
            model, task_type, task_data[task_type],
            attn_means, mlp_means, config
        )
        all_circuits[task_type] = circuit
        
        # Save individual circuit
        circuit_file = output_dir / f"circuit_{task_type}.json"
        with open(circuit_file, 'w') as f:
            json.dump(circuit, f, indent=2)
        print(f"  Saved: {circuit_file}")
    
    # Save summary
    summary = {
        'model_checkpoint': args.checkpoint,
        'model_config': args.config,
        'extraction_config': asdict(config),
        'tasks': tasks,
        'circuits': {},
    }
    for task, circ in all_circuits.items():
        summary['circuits'][task] = {
            'num_attention_channels': circ['num_attention_channels'],  # Gao node
            'num_mlp_neurons': circ['num_mlp_neurons'],
            'total_nodes': circ['total_nodes'],
            'total_possible': circ['total_possible'],
            'circuit_fraction': circ['circuit_fraction'],
            'full_model_accuracy': circ.get('full_model_accuracy', -1.0),
            'full_model_exact_match': circ.get('full_model_exact_match', -1.0),
            'circuit_accuracy': circ['circuit_accuracy'],
            'circuit_exact_match': circ.get('circuit_exact_match', -1.0),
            'exact_match_retention': circ.get('exact_match_retention', -1.0),
            'accuracy_retention': circ.get('accuracy_retention', -1.0),
            'circuit_edge_fraction': circ.get('circuit_edge_fraction', -1.0),
        }
    
    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    
    # Print summary table
    print(f"\n{'='*100}")
    print(f"Circuit Extraction Summary — Full Model vs Circuit Accuracy")
    print(f"{'='*100}")
    print(f"{'Task':<15} {'AttnCh':>8} {'MLP':>9} {'Total':>8} {'Frac':>7}  "
          f"{'Full EM':>8} {'Circ EM':>8} {'EM Ret':>8}")
    print(f"{'-'*15} {'-'*8} {'-'*9} {'-'*8} {'-'*7}  {'-'*8} {'-'*8} {'-'*8}")
    for task, circ in all_circuits.items():
        full_em = circ.get('full_model_exact_match', -1.0)
        circ_em = circ.get('circuit_exact_match', -1.0)
        em_ret  = circ.get('exact_match_retention', -1.0)
        flag    = '' if em_ret < 0 else ('✓' if em_ret > 0.85 else '⚠')
        print(f"{task:<15} {circ['num_attention_channels']:>8} {circ['num_mlp_neurons']:>9} "
              f"{circ['total_nodes']:>8} {circ['circuit_fraction']:>7.3f}  "
              f"{full_em:>8.3f} {circ_em:>8.3f} {em_ret:>7.3f} {flag}")

    # ── Extra threshold variants (post-hoc, no re-training) ───────────────
    # Threshold only changes the final binarization of the already-trained masks;
    # the optimization result is identical.  We re-use the saved mask z-values
    # (attn_mask_values / mlp_mask_values) to build circuits at every extra
    # threshold without paying any additional GPU time.
    if args.extra_thresholds:
        n_head = model.config.n_head
        d_head = model.config.d_head
        for extra_thr in sorted(set(args.extra_thresholds)):
            if abs(extra_thr - args.threshold) < 1e-9:
                continue  # identical to primary — skip

            thr_dir = output_dir.parent / f"circuits_thr{extra_thr}"
            thr_dir.mkdir(parents=True, exist_ok=True)

            thr_summary: dict = {
                'model_checkpoint': args.checkpoint,
                'model_config': args.config,
                'extraction_config': {**asdict(config), 'threshold': extra_thr,
                                      '_note': 'post-hoc rethreshold — no re-optimization'},
                'tasks': tasks,
                'threshold_variant_of': str(output_dir),
                'circuits': {},
            }

            for task, circ in all_circuits.items():
                extra_circ = rethreshold_circuit(circ, extra_thr, n_head, d_head)
                with open(thr_dir / f"circuit_{task}.json", 'w') as f:
                    json.dump(extra_circ, f, indent=2)
                thr_summary['circuits'][task] = {
                    'num_attention_channels':  extra_circ['num_attention_channels'],
                    'num_mlp_neurons':         extra_circ['num_mlp_neurons'],
                    'total_nodes':             extra_circ['total_nodes'],
                    'total_possible':          extra_circ['total_possible'],
                    'circuit_fraction':        extra_circ['circuit_fraction'],
                    'full_model_accuracy':     circ.get('full_model_accuracy', -1.0),
                    'full_model_exact_match':  circ.get('full_model_exact_match', -1.0),
                    # accuracy/em left -1.0: Phase 4 will evaluate via mean ablation
                    'circuit_accuracy':        -1.0,
                    'circuit_exact_match':     -1.0,
                    'exact_match_retention':   -1.0,
                    'accuracy_retention':      -1.0,
                    'circuit_edge_fraction':   -1.0,
                }

            with open(thr_dir / "extraction_summary.json", 'w') as f:
                json.dump(thr_summary, f, indent=2)

            fracs = [v['circuit_fraction'] for v in thr_summary['circuits'].values()]
            mean_frac = sum(fracs) / len(fracs) if fracs else 0.0
            print(f"\n[extra thr={extra_thr}] {len(all_circuits)} circuits → {thr_dir}/"
                  f"  (mean fraction={mean_frac:.3f})")


if __name__ == "__main__":
    main()
