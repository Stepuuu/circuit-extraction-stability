"""Loss-budget sweep for the fixed-loss pruning graph.

After fitting the continuous pruning scores, selects for each loss budget tau
the smallest top-k circuit whose ablated loss stays at or below tau, and
reports node fractions across budgets.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = str(ROOT)
CIRCUITS_DIR = ROOT / "circuits"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if str(CIRCUITS_DIR) not in sys.path:
    sys.path.insert(0, str(CIRCUITS_DIR))

from circuit_extraction import COMPOSITION_TASKS, TokenizerWrapper, load_model, prepare_task_data
from factorized_graph_core import (
    ATTN_QK_KEY,
    ATTN_V_KEY,
    MLP_NEURON_KEY,
    RESID_READ_ATTN_KEY,
    RESID_READ_MLP_KEY,
    RESID_WRITE_ATTN_KEY,
    RESID_WRITE_MLP_KEY,
    GaoAlignedAblationModel,
    annotate_gao_edge_stats,
    compute_gao_mean_activations,
)
from factorized_graph_core_qk import (
    ATTN_K_KEY as ATTN_K_KEY_V2,
    ATTN_Q_KEY as ATTN_Q_KEY_V2,
    ATTN_V_KEY as ATTN_V_KEY_V2,
    MLP_NEURON_KEY as MLP_NEURON_KEY_V2,
    RESID_READ_ATTN_KEY as RESID_READ_ATTN_KEY_V2,
    RESID_READ_MLP_KEY as RESID_READ_MLP_KEY_V2,
    RESID_WRITE_ATTN_KEY as RESID_WRITE_ATTN_KEY_V2,
    RESID_WRITE_MLP_KEY as RESID_WRITE_MLP_KEY_V2,
    GaoAlignedAblationModelV2,
    annotate_gao_edge_stats_v2,
)


V1_GROUPS = [
    RESID_READ_ATTN_KEY,
    ATTN_QK_KEY,
    ATTN_V_KEY,
    RESID_WRITE_ATTN_KEY,
    RESID_READ_MLP_KEY,
    MLP_NEURON_KEY,
    RESID_WRITE_MLP_KEY,
]
V2_GROUPS = [
    RESID_READ_ATTN_KEY_V2,
    ATTN_Q_KEY_V2,
    ATTN_K_KEY_V2,
    ATTN_V_KEY_V2,
    RESID_WRITE_ATTN_KEY_V2,
    RESID_READ_MLP_KEY_V2,
    MLP_NEURON_KEY_V2,
    RESID_WRITE_MLP_KEY_V2,
]

V1_MASK_KEYS = {
    RESID_READ_ATTN_KEY: "attn_read_mask_values",
    ATTN_QK_KEY: "attn_qk_mask_values",
    ATTN_V_KEY: "attn_v_mask_values",
    RESID_WRITE_ATTN_KEY: "attn_write_mask_values",
    RESID_READ_MLP_KEY: "mlp_read_mask_values",
    MLP_NEURON_KEY: "mlp_mask_values",
    RESID_WRITE_MLP_KEY: "mlp_write_mask_values",
}
V2_MASK_KEYS = {
    RESID_READ_ATTN_KEY_V2: "attn_read_mask_values",
    ATTN_Q_KEY_V2: "attn_q_mask_values",
    ATTN_K_KEY_V2: "attn_k_mask_values",
    ATTN_V_KEY_V2: "attn_v_mask_values",
    RESID_WRITE_ATTN_KEY_V2: "attn_write_mask_values",
    RESID_READ_MLP_KEY_V2: "mlp_read_mask_values",
    MLP_NEURON_KEY_V2: "mlp_mask_values",
    RESID_WRITE_MLP_KEY_V2: "mlp_write_mask_values",
}

COUNT_KEYS = {
    RESID_READ_ATTN_KEY: "num_attn_resid_reads",
    ATTN_QK_KEY: "num_attn_qk_channels",
    ATTN_V_KEY: "num_attn_v_channels",
    RESID_WRITE_ATTN_KEY: "num_attn_resid_writes",
    RESID_READ_MLP_KEY: "num_mlp_resid_reads",
    MLP_NEURON_KEY: "num_mlp_neurons",
    RESID_WRITE_MLP_KEY: "num_mlp_resid_writes",
    RESID_READ_ATTN_KEY_V2: "num_attn_resid_reads",
    ATTN_Q_KEY_V2: "num_attn_q_channels",
    ATTN_K_KEY_V2: "num_attn_k_channels",
    ATTN_V_KEY_V2: "num_attn_v_channels",
    RESID_WRITE_ATTN_KEY_V2: "num_attn_resid_writes",
    RESID_READ_MLP_KEY_V2: "num_mlp_resid_reads",
    MLP_NEURON_KEY_V2: "num_mlp_neurons",
    RESID_WRITE_MLP_KEY_V2: "num_mlp_resid_writes",
}


def _tag_loss(loss: float) -> str:
    return f"tau_{loss:.2f}".replace(".", "p")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _unpack_logits(output):
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def _evaluate_full_model(model, input_ids, target_ids, batch_size: int) -> dict[str, float]:
    n_samples = input_ids.size(0)
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    exact = 0
    for start in range(0, n_samples, batch_size):
        b_in = input_ids[start:start + batch_size]
        b_tgt = target_ids[start:start + batch_size]
        logits = _unpack_logits(model(b_in, b_tgt))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            b_tgt.reshape(-1),
            ignore_index=-1,
            reduction="sum",
        )
        mask = b_tgt != -1
        preds = logits.argmax(dim=-1)
        correct += (preds[mask] == b_tgt[mask]).sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item()
        for i in range(b_tgt.shape[0]):
            sm = mask[i]
            if sm.sum() > 0 and (preds[i][sm] == b_tgt[i][sm]).all():
                exact += 1
    return {
        "loss": total_loss / max(total_tokens, 1),
        "accuracy": correct / max(total_tokens, 1),
        "exact_match": exact / max(n_samples, 1),
    }


@torch.no_grad()
def _evaluate_fixed_circuit(ablation_model, circuit: dict[str, Any], input_ids, target_ids, batch_size: int) -> dict[str, float]:
    ablation_model.set_keep_circuit(circuit)
    n_samples = input_ids.size(0)
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    exact = 0
    for start in range(0, n_samples, batch_size):
        b_in = input_ids[start:start + batch_size]
        b_tgt = target_ids[start:start + batch_size]
        logits, _ = ablation_model(b_in, b_tgt)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            b_tgt.reshape(-1),
            ignore_index=-1,
            reduction="sum",
        )
        mask = b_tgt != -1
        preds = logits.argmax(dim=-1)
        correct += (preds[mask] == b_tgt[mask]).sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item()
        for i in range(b_tgt.shape[0]):
            sm = mask[i]
            if sm.sum() > 0 and (preds[i][sm] == b_tgt[i][sm]).all():
                exact += 1
    ablation_model.reset_masks()
    return {
        "loss": total_loss / max(total_tokens, 1),
        "accuracy": correct / max(total_tokens, 1),
        "exact_match": exact / max(n_samples, 1),
    }


def _score_entries(source: dict[str, Any], object_version: str, n_head: int) -> list[tuple[float, str, tuple[int, ...]]]:
    mask_keys = V2_MASK_KEYS if object_version == "v2" else V1_MASK_KEYS
    flat: list[tuple[float, str, tuple[int, ...]]] = []
    for group, mask_key in mask_keys.items():
        for layer_str, values in source[mask_key].items():
            layer = int(layer_str)
            if group in {ATTN_QK_KEY, ATTN_V_KEY, ATTN_Q_KEY_V2, ATTN_K_KEY_V2, ATTN_V_KEY_V2}:
                d_head = len(values) // n_head
                for idx, score in enumerate(values):
                    flat.append((float(score), group, (layer, idx // d_head, idx % d_head)))
            else:
                for idx, score in enumerate(values):
                    flat.append((float(score), group, (layer, idx)))
    flat.sort(key=lambda item: item[0], reverse=True)
    return flat


def _empty_circuit(source: dict[str, Any], object_version: str, include_mask_values: bool) -> dict[str, Any]:
    groups = V2_GROUPS if object_version == "v2" else V1_GROUPS
    mask_keys = V2_MASK_KEYS if object_version == "v2" else V1_MASK_KEYS
    circuit: dict[str, Any] = {group: [] for group in groups}
    if include_mask_values:
        for mask_key in mask_keys.values():
            circuit[mask_key] = copy.deepcopy(source[mask_key])
    circuit["node_schema"] = "gao_exact_pruning_v2" if object_version == "v2" else "gao_exact_pruning_v1"
    circuit["mask_formalism"] = source.get("mask_formalism", "heaviside_ste")
    return circuit


def _finalize_counts(circuit: dict[str, Any], object_version: str, total_possible: int) -> None:
    groups = V2_GROUPS if object_version == "v2" else V1_GROUPS
    for group in groups:
        circuit[COUNT_KEYS[group]] = len(circuit[group])
    circuit["total_nodes"] = sum(len(circuit[group]) for group in groups)
    circuit["total_possible"] = total_possible
    circuit["circuit_fraction"] = circuit["total_nodes"] / max(total_possible, 1)


def _circuit_from_topk(
    source: dict[str, Any],
    sorted_entries: list[tuple[float, str, tuple[int, ...]]],
    k: int,
    object_version: str,
    include_mask_values: bool = False,
) -> dict[str, Any]:
    circuit = _empty_circuit(source, object_version, include_mask_values)
    for _, group, idx in sorted_entries[:max(int(k), 0)]:
        circuit[group].append(idx)
    _finalize_counts(circuit, object_version, int(source.get("total_possible") or len(sorted_entries)))
    return circuit


def _select_for_target(
    source: dict[str, Any],
    sorted_entries: list[tuple[float, str, tuple[int, ...]]],
    object_version: str,
    target_loss: float,
    ablation_model,
    input_ids,
    target_ids,
    batch_size: int,
    eval_cache: dict[int, tuple[dict[str, Any], dict[str, float]]],
    candidate_ks: list[int] | None = None,
) -> tuple[int, dict[str, Any], dict[str, float], bool]:
    total_nodes = len(sorted_entries)

    def evaluate_k(k: int) -> tuple[dict[str, Any], dict[str, float]]:
        if k not in eval_cache:
            circuit = _circuit_from_topk(source, sorted_entries, k, object_version, include_mask_values=False)
            metrics = _evaluate_fixed_circuit(ablation_model, circuit, input_ids, target_ids, batch_size)
            eval_cache[k] = (circuit, metrics)
        return eval_cache[k]

    if candidate_ks is not None:
        best_loss_item: tuple[int, dict[str, Any], dict[str, float]] | None = None
        for k in candidate_ks:
            circuit, metrics = evaluate_k(k)
            if best_loss_item is None or metrics["loss"] < best_loss_item[2]["loss"]:
                best_loss_item = (k, circuit, metrics)
            if metrics["loss"] <= target_loss:
                final_circuit = _circuit_from_topk(source, sorted_entries, k, object_version, include_mask_values=True)
                return k, final_circuit, metrics, True
        if best_loss_item is None:
            raise RuntimeError("No candidate k values were available.")
        selected_k, _, metrics = best_loss_item
        final_circuit = _circuit_from_topk(source, sorted_entries, selected_k, object_version, include_mask_values=True)
        return selected_k, final_circuit, metrics, False

    low, high = 1, total_nodes
    best = None
    while low <= high:
        mid = (low + high) // 2
        circuit, metrics = evaluate_k(mid)
        if metrics["loss"] <= target_loss:
            best = (mid, circuit, metrics)
            high = mid - 1
        else:
            low = mid + 1

    target_satisfied = best is not None
    if best is None:
        circuit, metrics = evaluate_k(total_nodes)
        best = (total_nodes, circuit, metrics)

    selected_k, _, metrics = best
    final_circuit = _circuit_from_topk(source, sorted_entries, selected_k, object_version, include_mask_values=True)
    return selected_k, final_circuit, metrics, target_satisfied


def _candidate_ks(total_nodes: int, source_selected_k: int | None, grid_points: int) -> list[int]:
    ks = {1, total_nodes}
    grid_points = max(int(grid_points), 1)
    for i in range(grid_points + 1):
        ks.add(1 + round((total_nodes - 1) * i / grid_points))
    if source_selected_k is not None and 1 <= source_selected_k <= total_nodes:
        ks.add(source_selected_k)
        for factor in [0.05, 0.10, 0.20, 0.33, 0.50, 0.67, 0.75, 0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10, 1.25, 1.50, 2.00, 3.00]:
            ks.add(round(source_selected_k * factor))
        for offset in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
            ks.add(source_selected_k - offset)
            ks.add(source_selected_k + offset)
    return sorted(k for k in ks if 1 <= k <= total_nodes)


def _summary_row(circuit: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "total_nodes",
        "total_possible",
        "circuit_fraction",
        "edge_count",
        "total_possible_edges",
        "circuit_edge_fraction",
        "selected_k",
        "target_loss",
        "target_satisfied",
        "full_model_loss",
        "full_model_accuracy",
        "full_model_exact_match",
        "circuit_loss",
        "circuit_accuracy",
        "circuit_exact_match",
        "accuracy_retention",
        "exact_match_retention",
        "node_schema",
    ]
    return {key: circuit.get(key) for key in keys if key in circuit}


def run_tau_sweep(args: argparse.Namespace) -> None:
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
    source_dir = Path(args.source_circuits_dir)
    output_root = Path(args.output_root)
    target_losses = sorted(float(x) for x in args.target_losses)
    tasks = args.tasks or COMPOSITION_TASKS

    model = load_model(args.checkpoint, args.config, device)
    tokenizer_path = ROOT / "artifacts" / "tokenizer" / "trained" / "tokenizer.model"
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
    tokenizer = TokenizerWrapper(sp)

    first_source = _load_json(source_dir / f"circuit_{tasks[0]}.json")
    source_config = first_source.get("config", {})
    num_samples = args.num_samples or int(source_config.get("num_samples_per_task", 100))
    seed_base = args.seed_base if args.seed_base is not None else int(source_config.get("seed_base", 123_456_000))
    num_reference_samples = args.num_reference_samples or int(source_config.get("num_reference_samples", 200))
    reference_seed_base = (
        args.reference_seed_base
        if args.reference_seed_base is not None
        else int(source_config.get("reference_seed_base", 111_111_000))
    )
    supervision = args.supervision or str(source_config.get("supervision", "tactic"))
    batch_size = args.batch_size or int(source_config.get("batch_size", 32))
    mean_batch_size = args.mean_batch_size or batch_size

    task_data = prepare_task_data(tasks, num_samples, seed_base, tokenizer, model.config.block_size, device, supervision=supervision)
    mean_acts = compute_gao_mean_activations(
        model,
        tokenizer,
        num_reference_samples,
        reference_seed_base,
        model.config.block_size,
        device,
        batch_size=mean_batch_size,
    )

    ablation_cls = GaoAlignedAblationModelV2 if args.object_version == "v2" else GaoAlignedAblationModel
    annotate_fn = annotate_gao_edge_stats_v2 if args.object_version == "v2" else annotate_gao_edge_stats
    ablation_model = ablation_cls(model).to(device)
    ablation_model.set_mean_activations(
        mean_acts["attn_resid_reads"],
        mean_acts["q_means"],
        mean_acts["k_means"],
        mean_acts["attn_v_channels"],
        mean_acts["attn_resid_writes"],
        mean_acts["mlp_resid_reads"],
        mean_acts["mlp_neurons"],
        mean_acts["mlp_resid_writes"],
    )

    summaries: dict[float, dict[str, Any]] = {
        tau: {
            "model_checkpoint": str(args.checkpoint),
            "model_config": str(args.config),
            "node_schema": "gao_exact_pruning_v2" if args.object_version == "v2" else "gao_exact_pruning_v1",
            "object_version": args.object_version,
            "target_loss": tau,
            "source_circuits_dir": str(source_dir),
            "config": {
                "num_samples_per_task": num_samples,
                "seed_base": seed_base,
                "num_reference_samples": num_reference_samples,
                "reference_seed_base": reference_seed_base,
                "supervision": supervision,
                "batch_size": batch_size,
                "target_losses": target_losses,
                "rerank_from_saved_scores": True,
            },
            "circuits": {},
        }
        for tau in target_losses
    }

    for task in tasks:
        source_path = source_dir / f"circuit_{task}.json"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source = _load_json(source_path)
        sorted_entries = _score_entries(source, args.object_version, model.config.n_head)
        candidate_ks = _candidate_ks(len(sorted_entries), source.get("selected_k"), args.candidate_grid_points)
        input_ids = torch.stack([sample["input_ids"] for sample in task_data[task]]).to(device)
        target_ids = torch.stack([sample["target_ids"] for sample in task_data[task]]).to(device)
        full_metrics = _evaluate_full_model(model, input_ids, target_ids, batch_size)
        eval_cache: dict[int, tuple[dict[str, Any], dict[str, float]]] = {}

        for tau in target_losses:
            tau_dir = output_root / _tag_loss(tau) / "loss_threshold" / "circuits"
            circuit_out = tau_dir / f"circuit_{task}.json"
            if circuit_out.exists() and not args.force:
                existing = _load_json(circuit_out)
                summaries[tau]["circuits"][task] = _summary_row(existing)
                continue

            selected_k, circuit, circuit_metrics, target_satisfied = _select_for_target(
                source,
                sorted_entries,
                args.object_version,
                tau,
                ablation_model,
                input_ids,
                target_ids,
                batch_size,
                eval_cache,
                candidate_ks=candidate_ks,
            )
            annotate_fn(model, circuit)
            circuit["task_type"] = task
            circuit["selected_k"] = int(selected_k)
            circuit["target_loss"] = tau
            circuit["target_satisfied"] = bool(target_satisfied)
            circuit["source_target_loss"] = source.get("target_loss")
            circuit["source_selected_k"] = source.get("selected_k")
            circuit["source_circuit_loss"] = source.get("circuit_loss")
            circuit["candidate_grid_points"] = args.candidate_grid_points
            circuit["candidate_count"] = len(candidate_ks)
            circuit["full_model_loss"] = full_metrics["loss"]
            circuit["full_model_accuracy"] = full_metrics["accuracy"]
            circuit["full_model_exact_match"] = full_metrics["exact_match"]
            circuit["circuit_loss"] = circuit_metrics["loss"]
            circuit["circuit_accuracy"] = circuit_metrics["accuracy"]
            circuit["circuit_exact_match"] = circuit_metrics["exact_match"]
            circuit["accuracy_retention"] = circuit_metrics["accuracy"] / max(full_metrics["accuracy"], 1e-8)
            circuit["exact_match_retention"] = (
                circuit_metrics["exact_match"] / max(full_metrics["exact_match"], 1e-8)
                if full_metrics["exact_match"] > 0.01
                else 0.0
            )
            circuit["config"] = summaries[tau]["config"]
            _write_json(circuit_out, circuit)
            summaries[tau]["circuits"][task] = _summary_row(circuit)
            print(
                f"{args.object_version} {task} tau={tau:.2f} k={selected_k} "
                f"loss={circuit_metrics['loss']:.6f} satisfied={target_satisfied}",
                flush=True,
            )

    for tau, summary in summaries.items():
        summary_dir = output_root / _tag_loss(tau) / "loss_threshold" / "circuits"
        if len(summary["circuits"]) == len(tasks):
            _write_json(summary_dir / "extraction_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-select exact-pruning circuits for multiple fixed loss budgets.")
    parser.add_argument("--source-circuits-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--object-version", choices=["v1", "v2"], required=True)
    parser.add_argument("--target-losses", nargs="+", type=float, required=True)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--num-reference-samples", type=int, default=None)
    parser.add_argument("--reference-seed-base", type=int, default=None)
    parser.add_argument("--supervision", choices=["all", "answer", "tactic"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mean-batch-size", type=int, default=None)
    parser.add_argument("--candidate-grid-points", type=int, default=80)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_tau_sweep(args)


if __name__ == "__main__":
    main()
