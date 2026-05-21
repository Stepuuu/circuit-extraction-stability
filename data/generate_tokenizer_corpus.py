#!/usr/bin/env python3
"""
Generate a tokenizer training corpus with truly random variable names.

Rationale: a BPE tokenizer trained on a random-name corpus naturally yields
- frequent logic keywords -> whole-word tokens (And.intro, Or.inl, exact, ...)
- random variable names -> character-level subwords

which is exactly what we want:
- logical structure gets a compact encoding -> rules are easy to learn
- variable names are encoded at character level -> the model cannot memorize
  specific names and must learn positional matching instead
"""

import argparse
import random
import sys
import os
from pathlib import Path
from tqdm import tqdm
import json
from collections import Counter

sys.path.insert(0, os.path.abspath("."))

from data.logic_task_generator import TrulyRandomDatasetGenerator


def generate_corpus(
    n_samples: int = 1_000_000,
    output_dir: str = "artifacts/tokenizer",
    seed: int = 42,
    n_distractors: int = 2,
    show_stats: bool = True,
):
    """Generate large-scale truly random corpus for tokenizer training."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    corpus_file = output_path / "tokenizer_corpus.txt"
    stats_file = output_path / "corpus_stats.json"

    print("=" * 70)
    print("Truly Random Corpus Generation for Tokenizer Training")
    print("=" * 70)
    print(f"Target samples: {n_samples:,}")
    print(f"Distractors per sample: {n_distractors}")
    print(f"Output: {corpus_file}")
    print()

    task_types = [
        'S_and', 'S_or_l', 'S_or_r', 'S_id',
        'T_and_or', 'T_or_and', 'T_and_or_and',
        'T_nested_and', 'T_nested_or'
    ]

    task_counts = Counter()
    total_chars = 0
    all_var_lengths = []

    print("Generating corpus...")

    with open(corpus_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(n_samples), desc="Generating"):
            sample_seed = seed + i

            gen = TrulyRandomDatasetGenerator(
                seed=sample_seed,
                n_distractors=n_distractors,
            )

            rng = random.Random(sample_seed)
            task_type = rng.choice(task_types)
            task_counts[task_type] += 1

            sample = gen.generate_sample(task_type)
            input_text, output_text = sample.to_input_output()

            f.write(input_text + '\n')
            f.write(output_text + '\n')

            total_chars += len(input_text) + len(output_text)

            for prop in sample.props:
                all_var_lengths.append(len(prop))

    # Stats
    stats = {
        "n_samples": n_samples,
        "total_chars": total_chars,
        "task_distribution": dict(task_counts),
        "avg_var_length": sum(all_var_lengths) / len(all_var_lengths) if all_var_lengths else 0,
        "var_length_distribution": dict(Counter(all_var_lengths)),
        "seed": seed,
        "n_distractors": n_distractors,
        "type": "truly_random",
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    if show_stats:
        print()
        print("=" * 70)
        print("Corpus Statistics")
        print("=" * 70)
        print(f"Total samples: {n_samples:,}")
        print(f"Total characters: {total_chars:,}")
        print(f"Avg variable name length: {stats['avg_var_length']:.1f}")
        print()
        print("Task distribution:")
        for task, count in sorted(task_counts.items()):
            pct = count / n_samples * 100
            print(f"  {task:<20} {count:>8,} ({pct:>5.1f}%)")
        print()
        print(f"Saved: {corpus_file}")

    return corpus_file, stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate truly random corpus for tokenizer training"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1_000_000,
        help="Number of samples (default: 1,000,000)"
    )
    parser.add_argument("--output-dir", default="artifacts/tokenizer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-distractors", type=int, default=2)

    args = parser.parse_args()

    generate_corpus(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        n_distractors=args.n_distractors,
    )


if __name__ == "__main__":
    main()
