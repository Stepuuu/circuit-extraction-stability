#!/usr/bin/env python3
"""
Gao-aligned circuit analysis core utilities.

This module implements a more fine-grained node schema intended to better match
Gao et al. (2025):
  - attention residual reads
  - attention QK channels
  - attention V channels
  - attention residual writes
  - MLP residual reads
  - MLP neurons
  - MLP residual writes

The implementation is designed as a parallel path to the existing
`circuit_extraction.py` pipeline, so we can compare the same checkpoints under
different circuit formalisms without retraining.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_analysis_dir = os.path.dirname(os.path.abspath(__file__))
if _analysis_dir not in sys.path:
    sys.path.insert(0, _analysis_dir)

from circuit_extraction import HardConcreteMask


RESID_READ_ATTN_KEY = "attn_resid_reads"
ATTN_QK_KEY = "attn_qk_channels"
ATTN_V_KEY = "attn_v_channels"
RESID_WRITE_ATTN_KEY = "attn_resid_writes"
RESID_READ_MLP_KEY = "mlp_resid_reads"
MLP_NEURON_KEY = "mlp_neurons"
RESID_WRITE_MLP_KEY = "mlp_resid_writes"


def _apply_feature_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    mean: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply a broadcastable mask with mean ablation."""
    mask = mask.to(x.dtype)
    if mean is None:
        return x * mask
    return mask * x + (1 - mask) * mean.to(x.dtype)


def rethreshold_gao_circuit(
    circuit: dict,
    new_threshold: float,
    n_layer: int,
    n_head: int,
    d_head: int,
    d_model: int,
    d_mlp: int,
) -> dict:
    """Re-binarize a saved Gao-aligned circuit at a different threshold."""
    import copy

    new_circ = copy.deepcopy(circuit)
    new_circ[RESID_READ_ATTN_KEY] = []
    new_circ[ATTN_QK_KEY] = []
    new_circ[ATTN_V_KEY] = []
    new_circ[RESID_WRITE_ATTN_KEY] = []
    new_circ[RESID_READ_MLP_KEY] = []
    new_circ[MLP_NEURON_KEY] = []
    new_circ[RESID_WRITE_MLP_KEY] = []

    for l in range(n_layer):
        for c, v in enumerate(new_circ["attn_read_mask_values"][str(l)]):
            if v > new_threshold:
                new_circ[RESID_READ_ATTN_KEY].append((l, c))
        qk_vals = new_circ["attn_qk_mask_values"][str(l)]
        v_vals = new_circ["attn_v_mask_values"][str(l)]
        for h in range(n_head):
            for c in range(d_head):
                idx = h * d_head + c
                if qk_vals[idx] > new_threshold:
                    new_circ[ATTN_QK_KEY].append((l, h, c))
                if v_vals[idx] > new_threshold:
                    new_circ[ATTN_V_KEY].append((l, h, c))
        for c, v in enumerate(new_circ["attn_write_mask_values"][str(l)]):
            if v > new_threshold:
                new_circ[RESID_WRITE_ATTN_KEY].append((l, c))
        for c, v in enumerate(new_circ["mlp_read_mask_values"][str(l)]):
            if v > new_threshold:
                new_circ[RESID_READ_MLP_KEY].append((l, c))
        for n_idx, v in enumerate(new_circ["mlp_mask_values"][str(l)]):
            if v > new_threshold:
                new_circ[MLP_NEURON_KEY].append((l, n_idx))
        for c, v in enumerate(new_circ["mlp_write_mask_values"][str(l)]):
            if v > new_threshold:
                new_circ[RESID_WRITE_MLP_KEY].append((l, c))

    new_circ["num_attn_resid_reads"] = len(new_circ[RESID_READ_ATTN_KEY])
    new_circ["num_attn_qk_channels"] = len(new_circ[ATTN_QK_KEY])
    new_circ["num_attn_v_channels"] = len(new_circ[ATTN_V_KEY])
    new_circ["num_attn_resid_writes"] = len(new_circ[RESID_WRITE_ATTN_KEY])
    new_circ["num_mlp_resid_reads"] = len(new_circ[RESID_READ_MLP_KEY])
    new_circ["num_mlp_neurons"] = len(new_circ[MLP_NEURON_KEY])
    new_circ["num_mlp_resid_writes"] = len(new_circ[RESID_WRITE_MLP_KEY])
    new_circ["total_nodes"] = (
        new_circ["num_attn_resid_reads"]
        + new_circ["num_attn_qk_channels"]
        + new_circ["num_attn_v_channels"]
        + new_circ["num_attn_resid_writes"]
        + new_circ["num_mlp_resid_reads"]
        + new_circ["num_mlp_neurons"]
        + new_circ["num_mlp_resid_writes"]
    )
    new_circ["total_possible"] = 4 * n_layer * d_model + 2 * n_layer * n_head * d_head + n_layer * d_mlp
    new_circ["circuit_fraction"] = new_circ["total_nodes"] / max(new_circ["total_possible"], 1)
    for key in ("circuit_accuracy", "circuit_exact_match", "accuracy_retention", "exact_match_retention"):
        new_circ[key] = -1.0
    return new_circ


class _GaoAlignedBase(nn.Module):
    """Common forward logic for learnable and fixed-mask Gao-aligned models."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layer = model.config.n_layer
        self.n_head = model.config.n_head
        self.d_head = model.config.d_head
        self.d_mlp = model.config.d_mlp
        self.d_model = model.config.d_model

        self.attn_read_means: Dict[int, torch.Tensor] = {}
        self.q_means: Dict[int, torch.Tensor] = {}
        self.k_means: Dict[int, torch.Tensor] = {}
        self.v_means: Dict[int, torch.Tensor] = {}
        self.attn_write_means: Dict[int, torch.Tensor] = {}
        self.mlp_read_means: Dict[int, torch.Tensor] = {}
        self.mlp_neuron_means: Dict[int, torch.Tensor] = {}
        self.mlp_write_means: Dict[int, torch.Tensor] = {}

    def set_mean_activations(
        self,
        attn_read_means: Dict[int, torch.Tensor],
        q_means: Dict[int, torch.Tensor],
        k_means: Dict[int, torch.Tensor],
        v_means: Dict[int, torch.Tensor],
        attn_write_means: Dict[int, torch.Tensor],
        mlp_read_means: Dict[int, torch.Tensor],
        mlp_neuron_means: Dict[int, torch.Tensor],
        mlp_write_means: Dict[int, torch.Tensor],
    ) -> None:
        self.attn_read_means = attn_read_means
        self.q_means = q_means
        self.k_means = k_means
        self.v_means = v_means
        self.attn_write_means = attn_write_means
        self.mlp_read_means = mlp_read_means
        self.mlp_neuron_means = mlp_neuron_means
        self.mlp_write_means = mlp_write_means

    def total_nodes(self) -> int:
        return (
            4 * self.n_layer * self.d_model
            + 2 * self.n_layer * self.n_head * self.d_head
            + self.n_layer * self.d_mlp
        )

    def _mask_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        device = idx.device
        b, t = idx.size()
        masks = self._mask_tensors(device)

        tok_emb = self.model.transformer.wte(idx)
        pos_emb = self.model.transformer.wpe.weight[:t].unsqueeze(0)
        x = self.model.transformer.drop(tok_emb + pos_emb)

        for layer_idx, block in enumerate(self.model.transformer.h):
            # ---- Attention block ----
            x_norm = block.ln_1(x)
            read_mask = masks[RESID_READ_ATTN_KEY][layer_idx].view(1, 1, self.d_model)
            read_mean = self.attn_read_means.get(layer_idx)
            if read_mean is not None:
                read_mean = read_mean.view(1, 1, self.d_model).to(device).expand_as(x_norm)
            x_norm = _apply_feature_mask(x_norm, read_mask, read_mean)

            qkv = block.attn.c_attn(x_norm)
            q, k, v = qkv.split(self.n_head * self.d_head, dim=2)

            q = q.view(b, t, self.n_head, self.d_head).transpose(1, 2)
            k = k.view(b, t, self.n_head, self.d_head).transpose(1, 2)
            v = v.view(b, t, self.n_head, self.d_head).transpose(1, 2)

            qk_mask = masks[ATTN_QK_KEY][layer_idx].view(1, self.n_head, 1, self.d_head)
            q_mean = self.q_means.get(layer_idx)
            if q_mean is not None:
                q_mean = q_mean.view(1, self.n_head, 1, self.d_head).to(device).expand_as(q)
            q = _apply_feature_mask(q, qk_mask, q_mean)

            k_mean = self.k_means.get(layer_idx)
            if k_mean is not None:
                k_mean = k_mean.view(1, self.n_head, 1, self.d_head).to(device).expand_as(k)
            k = _apply_feature_mask(k, qk_mask, k_mean)

            v_mask = masks[ATTN_V_KEY][layer_idx].view(1, self.n_head, 1, self.d_head)
            v_mean = self.v_means.get(layer_idx)
            if v_mean is not None:
                v_mean = v_mean.view(1, self.n_head, 1, self.d_head).to(device).expand_as(v)
            v = _apply_feature_mask(v, v_mask, v_mean)

            scale = 1.0 / (self.d_head ** 0.5)
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=scale)
            y = y.transpose(1, 2).contiguous().view(b, t, self.n_head * self.d_head)
            y = block.attn.c_proj(y)
            y = block.attn.resid_dropout(y)

            write_mask = masks[RESID_WRITE_ATTN_KEY][layer_idx].view(1, 1, self.d_model)
            write_mean = self.attn_write_means.get(layer_idx)
            if write_mean is not None:
                write_mean = write_mean.view(1, 1, self.d_model).to(device).expand_as(y)
            y = _apply_feature_mask(y, write_mask, write_mean)
            x = x + y

            # ---- MLP block ----
            x_norm2 = block.ln_2(x)
            mlp_read_mask = masks[RESID_READ_MLP_KEY][layer_idx].view(1, 1, self.d_model)
            mlp_read_mean = self.mlp_read_means.get(layer_idx)
            if mlp_read_mean is not None:
                mlp_read_mean = mlp_read_mean.view(1, 1, self.d_model).to(device).expand_as(x_norm2)
            x_norm2 = _apply_feature_mask(x_norm2, mlp_read_mask, mlp_read_mean)

            h = block.mlp.c_fc(x_norm2)
            h = block.mlp.act_fn(h)
            mlp_neuron_mask = masks[MLP_NEURON_KEY][layer_idx].view(1, 1, self.d_mlp)
            mlp_neuron_mean = self.mlp_neuron_means.get(layer_idx)
            if mlp_neuron_mean is not None:
                mlp_neuron_mean = mlp_neuron_mean.view(1, 1, self.d_mlp).to(device).expand_as(h)
            h = _apply_feature_mask(h, mlp_neuron_mask, mlp_neuron_mean)

            y = block.mlp.c_proj(h)
            y = block.mlp.dropout(y)

            mlp_write_mask = masks[RESID_WRITE_MLP_KEY][layer_idx].view(1, 1, self.d_model)
            mlp_write_mean = self.mlp_write_means.get(layer_idx)
            if mlp_write_mean is not None:
                mlp_write_mean = mlp_write_mean.view(1, 1, self.d_model).to(device).expand_as(y)
            y = _apply_feature_mask(y, mlp_write_mask, mlp_write_mean)
            x = x + y

        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x) + self.model.final_logits_bias

        if self.model.config.enable_bigram_table:
            additional = F.embedding(idx, self.model.bigram_table, padding_idx=-1)
            logits = logits + additional.to(x.dtype)

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        else:
            loss = torch.zeros(1, device=device)

        return logits, loss


class GaoAlignedMaskedModel(_GaoAlignedBase):
    """Learnable-mask Gao-aligned extraction model."""

    def __init__(self, model, temperature: float = 2 / 3, stretch_lo: float = -0.1, stretch_hi: float = 1.1):
        super().__init__(model)
        self.attn_read_masks = HardConcreteMask(self.n_layer * self.d_model, temperature, stretch_lo, stretch_hi)
        self.attn_qk_masks = HardConcreteMask(self.n_layer * self.n_head * self.d_head, temperature, stretch_lo, stretch_hi)
        self.attn_v_masks = HardConcreteMask(self.n_layer * self.n_head * self.d_head, temperature, stretch_lo, stretch_hi)
        self.attn_write_masks = HardConcreteMask(self.n_layer * self.d_model, temperature, stretch_lo, stretch_hi)
        self.mlp_read_masks = HardConcreteMask(self.n_layer * self.d_model, temperature, stretch_lo, stretch_hi)
        self.mlp_neuron_masks = HardConcreteMask(self.n_layer * self.d_mlp, temperature, stretch_lo, stretch_hi)
        self.mlp_write_masks = HardConcreteMask(self.n_layer * self.d_model, temperature, stretch_lo, stretch_hi)

    def _mask_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            RESID_READ_ATTN_KEY: self.attn_read_masks().view(self.n_layer, self.d_model).to(device),
            ATTN_QK_KEY: self.attn_qk_masks().view(self.n_layer, self.n_head, self.d_head).to(device),
            ATTN_V_KEY: self.attn_v_masks().view(self.n_layer, self.n_head, self.d_head).to(device),
            RESID_WRITE_ATTN_KEY: self.attn_write_masks().view(self.n_layer, self.d_model).to(device),
            RESID_READ_MLP_KEY: self.mlp_read_masks().view(self.n_layer, self.d_model).to(device),
            MLP_NEURON_KEY: self.mlp_neuron_masks().view(self.n_layer, self.d_mlp).to(device),
            RESID_WRITE_MLP_KEY: self.mlp_write_masks().view(self.n_layer, self.d_model).to(device),
        }

    def l0_loss(self) -> torch.Tensor:
        return (
            self.attn_read_masks.l0_loss()
            + self.attn_qk_masks.l0_loss()
            + self.attn_v_masks.l0_loss()
            + self.attn_write_masks.l0_loss()
            + self.mlp_read_masks.l0_loss()
            + self.mlp_neuron_masks.l0_loss()
            + self.mlp_write_masks.l0_loss()
        )

    def get_circuit(self, threshold: float = 0.5) -> dict:
        circuit = {
            RESID_READ_ATTN_KEY: [],
            ATTN_QK_KEY: [],
            ATTN_V_KEY: [],
            RESID_WRITE_ATTN_KEY: [],
            RESID_READ_MLP_KEY: [],
            MLP_NEURON_KEY: [],
            RESID_WRITE_MLP_KEY: [],
            "attn_read_mask_values": {},
            "attn_qk_mask_values": {},
            "attn_v_mask_values": {},
            "attn_write_mask_values": {},
            "mlp_read_mask_values": {},
            "mlp_mask_values": {},
            "mlp_write_mask_values": {},
            "node_schema": "gao_aligned_v1",
        }

        attn_read_vals = self.attn_read_masks.forward(deterministic=True).view(self.n_layer, self.d_model)
        attn_qk_vals = self.attn_qk_masks.forward(deterministic=True).view(self.n_layer, self.n_head, self.d_head)
        attn_v_vals = self.attn_v_masks.forward(deterministic=True).view(self.n_layer, self.n_head, self.d_head)
        attn_write_vals = self.attn_write_masks.forward(deterministic=True).view(self.n_layer, self.d_model)
        mlp_read_vals = self.mlp_read_masks.forward(deterministic=True).view(self.n_layer, self.d_model)
        mlp_neuron_vals = self.mlp_neuron_masks.forward(deterministic=True).view(self.n_layer, self.d_mlp)
        mlp_write_vals = self.mlp_write_masks.forward(deterministic=True).view(self.n_layer, self.d_model)

        for l in range(self.n_layer):
            circuit["attn_read_mask_values"][str(l)] = attn_read_vals[l].tolist()
            circuit["attn_qk_mask_values"][str(l)] = attn_qk_vals[l].reshape(-1).tolist()
            circuit["attn_v_mask_values"][str(l)] = attn_v_vals[l].reshape(-1).tolist()
            circuit["attn_write_mask_values"][str(l)] = attn_write_vals[l].tolist()
            circuit["mlp_read_mask_values"][str(l)] = mlp_read_vals[l].tolist()
            circuit["mlp_mask_values"][str(l)] = mlp_neuron_vals[l].tolist()
            circuit["mlp_write_mask_values"][str(l)] = mlp_write_vals[l].tolist()

            circuit[RESID_READ_ATTN_KEY].extend([(l, i) for i, v in enumerate(attn_read_vals[l]) if v > threshold])
            for h in range(self.n_head):
                for c in range(self.d_head):
                    if attn_qk_vals[l, h, c] > threshold:
                        circuit[ATTN_QK_KEY].append((l, h, c))
                    if attn_v_vals[l, h, c] > threshold:
                        circuit[ATTN_V_KEY].append((l, h, c))
            circuit[RESID_WRITE_ATTN_KEY].extend([(l, i) for i, v in enumerate(attn_write_vals[l]) if v > threshold])
            circuit[RESID_READ_MLP_KEY].extend([(l, i) for i, v in enumerate(mlp_read_vals[l]) if v > threshold])
            circuit[MLP_NEURON_KEY].extend([(l, i) for i, v in enumerate(mlp_neuron_vals[l]) if v > threshold])
            circuit[RESID_WRITE_MLP_KEY].extend([(l, i) for i, v in enumerate(mlp_write_vals[l]) if v > threshold])

        circuit["num_attn_resid_reads"] = len(circuit[RESID_READ_ATTN_KEY])
        circuit["num_attn_qk_channels"] = len(circuit[ATTN_QK_KEY])
        circuit["num_attn_v_channels"] = len(circuit[ATTN_V_KEY])
        circuit["num_attn_resid_writes"] = len(circuit[RESID_WRITE_ATTN_KEY])
        circuit["num_mlp_resid_reads"] = len(circuit[RESID_READ_MLP_KEY])
        circuit["num_mlp_neurons"] = len(circuit[MLP_NEURON_KEY])
        circuit["num_mlp_resid_writes"] = len(circuit[RESID_WRITE_MLP_KEY])
        circuit["total_nodes"] = sum(
            circuit[k]
            for k in [
                "num_attn_resid_reads",
                "num_attn_qk_channels",
                "num_attn_v_channels",
                "num_attn_resid_writes",
                "num_mlp_resid_reads",
                "num_mlp_neurons",
                "num_mlp_resid_writes",
            ]
        )
        circuit["total_possible"] = self.total_nodes()
        circuit["circuit_fraction"] = circuit["total_nodes"] / max(circuit["total_possible"], 1)
        return circuit


class GaoAlignedAblationModel(_GaoAlignedBase):
    """Fixed-mask Gao-aligned ablation model."""

    def __init__(self, model):
        super().__init__(model)
        self.attn_read_keep = torch.ones(self.n_layer, self.d_model, dtype=torch.bool)
        self.attn_qk_keep = torch.ones(self.n_layer, self.n_head, self.d_head, dtype=torch.bool)
        self.attn_v_keep = torch.ones(self.n_layer, self.n_head, self.d_head, dtype=torch.bool)
        self.attn_write_keep = torch.ones(self.n_layer, self.d_model, dtype=torch.bool)
        self.mlp_read_keep = torch.ones(self.n_layer, self.d_model, dtype=torch.bool)
        self.mlp_neuron_keep = torch.ones(self.n_layer, self.d_mlp, dtype=torch.bool)
        self.mlp_write_keep = torch.ones(self.n_layer, self.d_model, dtype=torch.bool)

    def _mask_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            RESID_READ_ATTN_KEY: self.attn_read_keep.float().to(device),
            ATTN_QK_KEY: self.attn_qk_keep.float().to(device),
            ATTN_V_KEY: self.attn_v_keep.float().to(device),
            RESID_WRITE_ATTN_KEY: self.attn_write_keep.float().to(device),
            RESID_READ_MLP_KEY: self.mlp_read_keep.float().to(device),
            MLP_NEURON_KEY: self.mlp_neuron_keep.float().to(device),
            RESID_WRITE_MLP_KEY: self.mlp_write_keep.float().to(device),
        }

    def reset_masks(self) -> None:
        self.attn_read_keep.fill_(True)
        self.attn_qk_keep.fill_(True)
        self.attn_v_keep.fill_(True)
        self.attn_write_keep.fill_(True)
        self.mlp_read_keep.fill_(True)
        self.mlp_neuron_keep.fill_(True)
        self.mlp_write_keep.fill_(True)

    def _set_nodes(self, circuit: dict, value: bool) -> None:
        for l, c in circuit.get(RESID_READ_ATTN_KEY, []):
            self.attn_read_keep[l, c] = value
        for l, h, c in circuit.get(ATTN_QK_KEY, []):
            self.attn_qk_keep[l, h, c] = value
        for l, h, c in circuit.get(ATTN_V_KEY, []):
            self.attn_v_keep[l, h, c] = value
        for l, c in circuit.get(RESID_WRITE_ATTN_KEY, []):
            self.attn_write_keep[l, c] = value
        for l, c in circuit.get(RESID_READ_MLP_KEY, []):
            self.mlp_read_keep[l, c] = value
        for l, n in circuit.get(MLP_NEURON_KEY, []):
            self.mlp_neuron_keep[l, n] = value
        for l, c in circuit.get(RESID_WRITE_MLP_KEY, []):
            self.mlp_write_keep[l, c] = value

    def set_keep_circuit(self, circuit: dict) -> None:
        self.attn_read_keep.fill_(False)
        self.attn_qk_keep.fill_(False)
        self.attn_v_keep.fill_(False)
        self.attn_write_keep.fill_(False)
        self.mlp_read_keep.fill_(False)
        self.mlp_neuron_keep.fill_(False)
        self.mlp_write_keep.fill_(False)
        self._set_nodes(circuit, True)

    def set_ablate_circuit(self, circuit: dict) -> None:
        self.reset_masks()
        self._set_nodes(circuit, False)

    def set_keep_union(self, circuits: List[dict]) -> None:
        self.attn_read_keep.fill_(False)
        self.attn_qk_keep.fill_(False)
        self.attn_v_keep.fill_(False)
        self.attn_write_keep.fill_(False)
        self.mlp_read_keep.fill_(False)
        self.mlp_neuron_keep.fill_(False)
        self.mlp_write_keep.fill_(False)
        for circuit in circuits:
            self._set_nodes(circuit, True)


@torch.no_grad()
def compute_gao_mean_activations(
    model,
    tokenizer,
    num_samples: int,
    seed_base: int,
    block_size: int,
    device: str,
    batch_size: int = 16,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Compute mean activations for all Gao-aligned node types."""
    from circuit_extraction import TASK_TYPES, generate_sample

    model.eval()
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    d_head = model.config.d_head
    d_mlp = model.config.d_mlp
    d_model = model.config.d_model

    attn_read_sums = {l: torch.zeros(d_model, device=device) for l in range(n_layer)}
    q_sums = {l: torch.zeros(n_head, d_head, device=device) for l in range(n_layer)}
    k_sums = {l: torch.zeros(n_head, d_head, device=device) for l in range(n_layer)}
    v_sums = {l: torch.zeros(n_head, d_head, device=device) for l in range(n_layer)}
    attn_write_sums = {l: torch.zeros(d_model, device=device) for l in range(n_layer)}
    mlp_read_sums = {l: torch.zeros(d_model, device=device) for l in range(n_layer)}
    mlp_neuron_sums = {l: torch.zeros(d_mlp, device=device) for l in range(n_layer)}
    mlp_write_sums = {l: torch.zeros(d_model, device=device) for l in range(n_layer)}
    count = 0

    all_inputs = []
    seed = seed_base
    for _ in range(num_samples):
        for task_type in TASK_TYPES:
            try:
                input_text, output_text = generate_sample(task_type, seed)
                tokens = tokenizer.encode(input_text + output_text)
                if len(tokens) <= block_size:
                    padded = tokens + [0] * (block_size - len(tokens))
                    all_inputs.append(torch.tensor(padded[:block_size], dtype=torch.long))
            except Exception:
                pass
            seed += 1

    for batch_start in range(0, len(all_inputs), batch_size):
        batch = torch.stack(all_inputs[batch_start:batch_start + batch_size]).to(device)
        b, t = batch.size()

        tok_emb = model.transformer.wte(batch)
        pos_emb = model.transformer.wpe.weight[:t].unsqueeze(0)
        x = model.transformer.drop(tok_emb + pos_emb)

        for l, block in enumerate(model.transformer.h):
            x_norm = block.ln_1(x)
            attn_read_sums[l] += x_norm.mean(dim=(0, 1))

            qkv = block.attn.c_attn(x_norm)
            q, k, v = qkv.split(n_head * d_head, dim=2)
            q = q.view(b, t, n_head, d_head).transpose(1, 2)
            k = k.view(b, t, n_head, d_head).transpose(1, 2)
            v = v.view(b, t, n_head, d_head).transpose(1, 2)
            q_sums[l] += q.mean(dim=(0, 2))
            k_sums[l] += k.mean(dim=(0, 2))
            v_sums[l] += v.mean(dim=(0, 2))

            scale = 1.0 / (d_head ** 0.5)
            y_attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=scale)
            y_attn = y_attn.transpose(1, 2).contiguous().view(b, t, n_head * d_head)
            y_proj = block.attn.c_proj(y_attn)
            y_proj = block.attn.resid_dropout(y_proj)
            attn_write_sums[l] += y_proj.mean(dim=(0, 1))
            x = x + y_proj

            x_norm2 = block.ln_2(x)
            mlp_read_sums[l] += x_norm2.mean(dim=(0, 1))
            h = block.mlp.c_fc(x_norm2)
            h = block.mlp.act_fn(h)
            mlp_neuron_sums[l] += h.mean(dim=(0, 1))
            y_mlp = block.mlp.c_proj(h)
            y_mlp = block.mlp.dropout(y_mlp)
            mlp_write_sums[l] += y_mlp.mean(dim=(0, 1))
            x = x + y_mlp

        count += 1

    return {
        RESID_READ_ATTN_KEY: {l: attn_read_sums[l] / count for l in range(n_layer)},
        "q_means": {l: q_sums[l] / count for l in range(n_layer)},
        "k_means": {l: k_sums[l] / count for l in range(n_layer)},
        ATTN_V_KEY: {l: v_sums[l] / count for l in range(n_layer)},
        RESID_WRITE_ATTN_KEY: {l: attn_write_sums[l] / count for l in range(n_layer)},
        RESID_READ_MLP_KEY: {l: mlp_read_sums[l] / count for l in range(n_layer)},
        MLP_NEURON_KEY: {l: mlp_neuron_sums[l] / count for l in range(n_layer)},
        RESID_WRITE_MLP_KEY: {l: mlp_write_sums[l] / count for l in range(n_layer)},
    }


def annotate_gao_edge_stats(model, circuit: dict) -> dict:
    """Compute exact edge counts for the selected Gao-aligned circuit."""
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    d_head = model.config.d_head
    d_model = model.config.d_model
    d_mlp = model.config.d_mlp
    nhd = n_head * d_head

    if not hasattr(model, "_gao_total_nonzero_edges"):
        total = 0
        for block in model.transformer.h:
            w_attn = block.attn.c_attn.weight.data
            q_w = w_attn[:nhd]
            k_w = w_attn[nhd:2 * nhd]
            v_w = w_attn[2 * nhd:3 * nhd]
            total += torch.count_nonzero(q_w).item()
            total += torch.count_nonzero(k_w).item()
            total += torch.count_nonzero(v_w).item()
            total += torch.count_nonzero(block.attn.c_proj.weight.data).item()
            total += torch.count_nonzero(block.mlp.c_fc.weight.data).item()
            total += torch.count_nonzero(block.mlp.c_proj.weight.data).item()
        model._gao_total_nonzero_edges = total

    selected = {
        RESID_READ_ATTN_KEY: {l: torch.zeros(d_model, dtype=torch.bool) for l in range(n_layer)},
        ATTN_QK_KEY: {l: torch.zeros(n_head, d_head, dtype=torch.bool) for l in range(n_layer)},
        ATTN_V_KEY: {l: torch.zeros(n_head, d_head, dtype=torch.bool) for l in range(n_layer)},
        RESID_WRITE_ATTN_KEY: {l: torch.zeros(d_model, dtype=torch.bool) for l in range(n_layer)},
        RESID_READ_MLP_KEY: {l: torch.zeros(d_model, dtype=torch.bool) for l in range(n_layer)},
        MLP_NEURON_KEY: {l: torch.zeros(d_mlp, dtype=torch.bool) for l in range(n_layer)},
        RESID_WRITE_MLP_KEY: {l: torch.zeros(d_model, dtype=torch.bool) for l in range(n_layer)},
    }

    for l, c in circuit.get(RESID_READ_ATTN_KEY, []):
        selected[RESID_READ_ATTN_KEY][l][c] = True
    for l, h, c in circuit.get(ATTN_QK_KEY, []):
        selected[ATTN_QK_KEY][l][h, c] = True
    for l, h, c in circuit.get(ATTN_V_KEY, []):
        selected[ATTN_V_KEY][l][h, c] = True
    for l, c in circuit.get(RESID_WRITE_ATTN_KEY, []):
        selected[RESID_WRITE_ATTN_KEY][l][c] = True
    for l, c in circuit.get(RESID_READ_MLP_KEY, []):
        selected[RESID_READ_MLP_KEY][l][c] = True
    for l, n in circuit.get(MLP_NEURON_KEY, []):
        selected[MLP_NEURON_KEY][l][n] = True
    for l, c in circuit.get(RESID_WRITE_MLP_KEY, []):
        selected[RESID_WRITE_MLP_KEY][l][c] = True

    breakdown = {
        "attn_read_to_q": 0,
        "attn_read_to_k": 0,
        "attn_read_to_v": 0,
        "attn_v_to_write": 0,
        "mlp_read_to_neuron": 0,
        "mlp_neuron_to_write": 0,
    }

    for l, block in enumerate(model.transformer.h):
        attn_read_sel = selected[RESID_READ_ATTN_KEY][l]
        qk_sel = selected[ATTN_QK_KEY][l].reshape(-1)
        v_sel = selected[ATTN_V_KEY][l].reshape(-1)
        attn_write_sel = selected[RESID_WRITE_ATTN_KEY][l]
        mlp_read_sel = selected[RESID_READ_MLP_KEY][l]
        mlp_neuron_sel = selected[MLP_NEURON_KEY][l]
        mlp_write_sel = selected[RESID_WRITE_MLP_KEY][l]

        w_attn = block.attn.c_attn.weight.data
        q_w = w_attn[:nhd]
        k_w = w_attn[nhd:2 * nhd]
        v_w = w_attn[2 * nhd:3 * nhd]
        proj_w = block.attn.c_proj.weight.data
        fc_w = block.mlp.c_fc.weight.data
        mlp_proj_w = block.mlp.c_proj.weight.data

        if attn_read_sel.any() and qk_sel.any():
            breakdown["attn_read_to_q"] += torch.count_nonzero(q_w[qk_sel][:, attn_read_sel]).item()
            breakdown["attn_read_to_k"] += torch.count_nonzero(k_w[qk_sel][:, attn_read_sel]).item()
        if attn_read_sel.any() and v_sel.any():
            breakdown["attn_read_to_v"] += torch.count_nonzero(v_w[v_sel][:, attn_read_sel]).item()
        if attn_write_sel.any() and v_sel.any():
            breakdown["attn_v_to_write"] += torch.count_nonzero(proj_w[attn_write_sel][:, v_sel]).item()
        if mlp_read_sel.any() and mlp_neuron_sel.any():
            breakdown["mlp_read_to_neuron"] += torch.count_nonzero(fc_w[mlp_neuron_sel][:, mlp_read_sel]).item()
        if mlp_write_sel.any() and mlp_neuron_sel.any():
            breakdown["mlp_neuron_to_write"] += torch.count_nonzero(mlp_proj_w[mlp_write_sel][:, mlp_neuron_sel]).item()

    edge_count = sum(breakdown.values())
    circuit["edge_count"] = edge_count
    circuit["total_possible_edges"] = model._gao_total_nonzero_edges
    circuit["circuit_edge_fraction"] = edge_count / max(model._gao_total_nonzero_edges, 1)
    circuit["edge_breakdown"] = breakdown
    return circuit
