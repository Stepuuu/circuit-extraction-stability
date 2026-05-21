"""Extraction driver for the factorized query/key graph objects.

Runs the factorized-Q/K extraction over checkpoints and writes per-task
circuit summaries (node schema: gao_aligned_v2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm
import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)
_analysis_dir = os.path.dirname(os.path.abspath(__file__))
if _analysis_dir not in sys.path:
    sys.path.insert(0, _analysis_dir)

from circuit_extraction import TASK_TYPES, SIMPLE_TASKS, COMPOSITION_TASKS, TokenizerWrapper, prepare_task_data, load_model
from factorized_graph_core_qk import GaoAlignedMaskedModelV2, compute_gao_mean_activations, annotate_gao_edge_stats_v2


@dataclass
class GaoAlignedExtractionConfigV2:
    mask_lr: float = 0.1
    lambda_l0: float = 0.02
    num_epochs: int = 150
    temperature: float = 2 / 3
    stretch_lo: float = -0.1
    stretch_hi: float = 1.1
    num_samples_per_task: int = 100
    seed_base: int = 123_456_000
    threshold: float = 0.5
    supervision: str = "tactic"
    num_reference_samples: int = 200
    reference_seed_base: int = 111_111_000
    device: str = "cuda"
    batch_size: int = 8
    mean_batch_size: int = 2


def _select_task_types(task_group: str, tasks: List[str] | None) -> List[str]:
    if tasks:
        return tasks
    if task_group == "simple":
        return SIMPLE_TASKS
    if task_group == "composition":
        return COMPOSITION_TASKS
    return TASK_TYPES


def _evaluate_exact_match(model, input_ids, target_ids, batch_size):
    n_samples = input_ids.size(0)
    exact = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            b_in = input_ids[start:start + batch_size]
            b_tgt = target_ids[start:start + batch_size]
            out = model(b_in, b_tgt)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(dim=-1)
            mask = b_tgt != -1
            correct += (preds[mask] == b_tgt[mask]).sum().item()
            total += mask.sum().item()
            for i in range(b_tgt.shape[0]):
                sm = mask[i]
                if sm.sum() > 0 and (preds[i][sm] == b_tgt[i][sm]).all():
                    exact += 1
    return correct / max(total, 1), exact / max(n_samples, 1)


def extract_circuit_for_task(model, task_type: str, task_data: List[Dict], mean_acts: Dict[str, Dict[int, torch.Tensor]], config: GaoAlignedExtractionConfigV2) -> dict:
    device = config.device
    masked_model = GaoAlignedMaskedModelV2(model, temperature=config.temperature, stretch_lo=config.stretch_lo, stretch_hi=config.stretch_hi).to(device)
    masked_model.set_mean_activations(
        mean_acts["attn_resid_reads"],
        mean_acts["q_means"],
        mean_acts["k_means"],
        mean_acts["attn_v_channels"],
        mean_acts["attn_resid_writes"],
        mean_acts["mlp_resid_reads"],
        mean_acts["mlp_neurons"],
        mean_acts["mlp_resid_writes"],
    )

    params = [
        *masked_model.attn_read_masks.parameters(),
        *masked_model.attn_q_masks.parameters(),
        *masked_model.attn_k_masks.parameters(),
        *masked_model.attn_v_masks.parameters(),
        *masked_model.attn_write_masks.parameters(),
        *masked_model.mlp_read_masks.parameters(),
        *masked_model.mlp_neuron_masks.parameters(),
        *masked_model.mlp_write_masks.parameters(),
    ]
    optimizer = torch.optim.Adam(params, lr=config.mask_lr)

    input_ids = torch.stack([s["input_ids"] for s in task_data]).to(device)
    target_ids = torch.stack([s["target_ids"] for s in task_data]).to(device)
    n_samples = len(task_data)
    total_nodes = masked_model.total_nodes()

    full_acc, full_em = _evaluate_exact_match(model, input_ids, target_ids, config.batch_size)
    best_em = -1.0
    best_loss = float("inf")
    best_state = None

    for epoch in range(config.num_epochs):
        masked_model.train()
        indices = torch.randperm(n_samples)
        total_task_loss = 0.0
        total_l0 = 0.0
        for start in range(0, n_samples, config.batch_size):
            batch_idx = indices[start:start + config.batch_size]
            b_in = input_ids[batch_idx]
            b_tgt = target_ids[batch_idx]
            _, task_loss = masked_model(b_in, b_tgt)
            l0 = masked_model.l0_loss() / total_nodes
            loss = task_loss + config.lambda_l0 * l0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_task_loss += task_loss.item()
            total_l0 += l0.item()

        denom = max(1, n_samples // config.batch_size)
        avg_task_loss = total_task_loss / denom
        avg_l0 = total_l0 / denom
        if epoch % 5 == 0 or epoch == config.num_epochs - 1:
            masked_model.eval()
            _, epoch_em = _evaluate_exact_match(masked_model, input_ids, target_ids, config.batch_size)
            combined = avg_task_loss + config.lambda_l0 * avg_l0
            if epoch_em > best_em or (epoch_em == best_em and combined < best_loss):
                best_em = epoch_em
                best_loss = combined
                best_state = {k: v.state_dict() for k, v in {
                    "attn_read_masks": masked_model.attn_read_masks,
                    "attn_q_masks": masked_model.attn_q_masks,
                    "attn_k_masks": masked_model.attn_k_masks,
                    "attn_v_masks": masked_model.attn_v_masks,
                    "attn_write_masks": masked_model.attn_write_masks,
                    "mlp_read_masks": masked_model.mlp_read_masks,
                    "mlp_neuron_masks": masked_model.mlp_neuron_masks,
                    "mlp_write_masks": masked_model.mlp_write_masks,
                }.items()}

    if best_state:
        for name, module in {
            "attn_read_masks": masked_model.attn_read_masks,
            "attn_q_masks": masked_model.attn_q_masks,
            "attn_k_masks": masked_model.attn_k_masks,
            "attn_v_masks": masked_model.attn_v_masks,
            "attn_write_masks": masked_model.attn_write_masks,
            "mlp_read_masks": masked_model.mlp_read_masks,
            "mlp_neuron_masks": masked_model.mlp_neuron_masks,
            "mlp_write_masks": masked_model.mlp_write_masks,
        }.items():
            module.load_state_dict(best_state[name])

    masked_model.eval()
    circuit = masked_model.get_circuit(config.threshold)
    circuit_acc, circuit_em = _evaluate_exact_match(masked_model, input_ids, target_ids, config.batch_size)
    circuit["task_type"] = task_type
    circuit["full_model_accuracy"] = full_acc
    circuit["full_model_exact_match"] = full_em
    circuit["circuit_accuracy"] = circuit_acc
    circuit["circuit_exact_match"] = circuit_em
    circuit["accuracy_retention"] = circuit_acc / max(full_acc, 1e-8)
    circuit["exact_match_retention"] = circuit_em / max(full_em, 1e-8) if full_em > 0.01 else 0.0
    circuit["config"] = asdict(config)
    annotate_gao_edge_stats_v2(model, circuit)
    return circuit


def main():
    parser = argparse.ArgumentParser(description="Factorized circuit extraction with separate query/key support objects")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--task-group", choices=["simple", "composition", "all"], default="all")
    parser.add_argument("--lambda-l0", type=float, default=0.02)
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--mask-lr", type=float, default=0.1)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--supervision", type=str, default="tactic", choices=["all", "answer", "tactic"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mean-batch-size", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    circuits_dir = output_dir / "circuits"
    circuits_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.config, device)
    tokenizer_path = os.path.join(_project_root, "artifacts/tokenizer/trained/tokenizer.model")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    tokenizer = TokenizerWrapper(sp)

    config = GaoAlignedExtractionConfigV2(
        lambda_l0=args.lambda_l0,
        num_epochs=args.num_epochs,
        mask_lr=args.mask_lr,
        num_samples_per_task=args.num_samples,
        threshold=args.threshold,
        supervision=args.supervision,
        device=device,
        batch_size=args.batch_size,
        mean_batch_size=args.mean_batch_size,
    )

    task_types = _select_task_types(args.task_group, args.tasks)
    task_data = prepare_task_data(task_types, config.num_samples_per_task, config.seed_base, tokenizer, model.config.block_size, device, supervision=config.supervision)
    mean_acts = compute_gao_mean_activations(model, tokenizer, config.num_reference_samples, config.reference_seed_base, model.config.block_size, device, batch_size=config.mean_batch_size)

    summary = {"model_checkpoint": args.checkpoint, "model_config": args.config, "node_schema": "gao_aligned_v2", "circuits": {}}
    for task_type in task_types:
        circuit = extract_circuit_for_task(model, task_type, task_data[task_type], mean_acts, config)
        (circuits_dir / f"circuit_{task_type}.json").write_text(json.dumps(circuit, indent=2), encoding="utf-8")
        summary["circuits"][task_type] = {
            "circuit_fraction": circuit["circuit_fraction"],
            "circuit_accuracy": circuit["circuit_accuracy"],
            "circuit_exact_match": circuit["circuit_exact_match"],
            "node_schema": circuit["node_schema"],
        }
        print(f"{task_type}: frac={circuit['circuit_fraction']:.3f} em={circuit['circuit_exact_match']:.3f}")
    summary["config"] = asdict(config)
    (circuits_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(circuits_dir)


if __name__ == "__main__":
    main()
