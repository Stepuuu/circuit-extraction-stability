"""Surface-form family variants of the logic-task generator.

Wraps the truly-random generator and, via the LOGIC_GENERATOR_FAMILY
environment variable, switches between the narrow baseline family and the
expanded families (broad_medium, broad_full), whose variants cover goal-first,
long-name, compact, and relabeled surface forms.
"""
from __future__ import annotations

import os
import random
import re
from typing import Tuple

from data.logic_task_generator import LogicSample, TrulyRandomDatasetGenerator


GENERATOR_FAMILY_ENV = "LOGIC_GENERATOR_FAMILY"


def current_generator_family(default: str = "narrow") -> str:
    return os.environ.get(GENERATOR_FAMILY_ENV, default)


def _replace_tokenwise(text: str, mapping: dict[str, str]) -> str:
    items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    for src, dst in items:
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(src)}(?![A-Za-z0-9_])", dst, text)
    return text


def _long_name_mapping(sample: LogicSample) -> dict[str, str]:
    mapping = {}
    for i, p in enumerate(sample.props):
        mapping[p] = f"prop_{i}_{p}_long"
    hyp_names = [h for h, _ in sample.premises + sample.distractors]
    for i, h in enumerate(hyp_names):
        mapping[h] = f"hyp_{i}_{h}_long"
    return mapping


def _format_standard(sample: LogicSample) -> Tuple[str, str]:
    return sample.to_input_output()


def _format_compact_input(sample: LogicSample) -> Tuple[str, str]:
    all_premises = sample.premises + sample.distractors
    prop_decl = " ".join(sorted(set(sample.props + [p for _, p in sample.distractors]))) + " : Prop"
    premise_str = " ; ".join(f"{h} : {p}" for h, p in all_premises)
    input_text = f"state_0 | {prop_decl} | {premise_str} | goal {sample.goal}"
    _, output_text = sample.to_input_output()
    return input_text, output_text


def _format_relabel_input(sample: LogicSample) -> Tuple[str, str]:
    all_premises = sample.premises + sample.distractors
    prop_decl = " ".join(sorted(set(sample.props + [p for _, p in sample.distractors]))) + " : Prop"
    premise_lines = [f"{h} : {p}" for h, p in all_premises]
    input_text = "theorem_0:\n" + prop_decl + "\nassumptions:\n" + "\n".join(premise_lines) + f"\nprove {sample.goal}"
    _, output_text = sample.to_input_output()
    return input_text, output_text


def _format_goal_first(sample: LogicSample) -> Tuple[str, str]:
    all_premises = sample.premises + sample.distractors
    prop_decl = " ".join(sorted(set(sample.props + [p for _, p in sample.distractors]))) + " : Prop"
    premise_lines = [f"{h} : {p}" for h, p in all_premises]
    input_text = f"state_0:\n⊢ {sample.goal}\nwhere\n{prop_decl}\n" + "\n".join(premise_lines)
    _, output_text = sample.to_input_output()
    return input_text, output_text


def _format_long_names(sample: LogicSample) -> Tuple[str, str]:
    inp, out = sample.to_input_output()
    mapping = _long_name_mapping(sample)
    return _replace_tokenwise(inp, mapping), _replace_tokenwise(out, mapping)


def _choose_variant(rng: random.Random, family: str) -> str:
    if family == "narrow":
        return "base"
    if family == "broad_medium":
        return rng.choices(
            ["base", "distractor_heavy", "goal_first_input", "long_names"],
            weights=[0.40, 0.20, 0.20, 0.20],
        )[0]
    if family == "broad_full":
        return rng.choices(
            ["base", "distractor_heavy", "compact_input", "relabel_input", "goal_first_input", "long_names"],
            weights=[0.20, 0.20, 0.15, 0.15, 0.15, 0.15],
        )[0]
    raise ValueError(f"Unknown generator family: {family}")


def generate_sample(task_type: str, seed: int, n_distractors: int = 2, family: str | None = None) -> Tuple[str, str]:
    family = family or current_generator_family()
    rng = random.Random(seed)
    variant = _choose_variant(rng, family)
    distractors = 6 if variant == "distractor_heavy" else n_distractors
    gen = TrulyRandomDatasetGenerator(seed=seed, n_distractors=distractors)
    sample = gen.generate_sample(task_type)

    if variant in ("base", "distractor_heavy"):
        return _format_standard(sample)
    if variant == "compact_input":
        return _format_compact_input(sample)
    if variant == "relabel_input":
        return _format_relabel_input(sample)
    if variant == "goal_first_input":
        return _format_goal_first(sample)
    if variant == "long_names":
        return _format_long_names(sample)
    raise ValueError(variant)
