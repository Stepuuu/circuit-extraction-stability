#!/usr/bin/env python3
"""
Train a BPE tokenizer for the synthetic Lean tactic-prediction tasks.

Trains on the ~1M-sample generated corpus, targeting dynamic (randomized)
variable names.
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime
import sentencepiece as spm


def train_tokenizer(
    corpus_file: str = "artifacts/tokenizer/tokenizer_corpus.txt",
    output_dir: str = "artifacts/tokenizer/trained",
    vocab_size: int = 2048,
    model_type: str = "bpe",
):
    """Train SentencePiece tokenizer on dynamic corpus."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🔤 Training Dynamic Tokenizer")
    print("=" * 70)
    print(f"Corpus: {corpus_file}")
    print(f"Vocab size: {vocab_size}")
    print(f"Model type: {model_type}")
    print(f"Output: {output_dir}")
    print()

    # Check corpus exists
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_file}\n"
            "Run data/generate_tokenizer_corpus.py first!"
        )

    # Get corpus stats
    with open(corpus_file, 'r') as f:
        line_count = sum(1 for _ in f)
    print(f"Corpus lines: {line_count:,}")
    print()

    # Train SentencePiece
    model_prefix = str(output_path / "tokenizer")

    print("Training SentencePiece model...")
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Key settings
        split_by_whitespace=False,  # allow whitespace to be encoded
        split_by_number=True,  # split on digits (important: keeps numeric suffixes separate)
        split_by_unicode_script=True,
        byte_fallback=True,  # fall back to byte-level encoding
        max_sentence_length=16384,
        normalization_rule_name='identity',  # keep original characters
        # Control rare tokens
        input_sentence_size=2000000,  # number of sampled sentences
        shuffle_input_sentence=True,
    )

    print(f"✅ Model saved: {model_prefix}.model")
    print(f"✅ Vocab saved: {model_prefix}.vocab")

    # Load and test
    print()
    print("=" * 70)
    print("🧪 Testing Tokenizer")
    print("=" * 70)

    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    # Test cases covering various variable-name patterns
    test_cases = [
        # Dynamic variable names
        "state_0:\nuser_id_732 alpha_45 x_1_567 : Prop",
        "h1 : flibble_123 ∧ glorp_456",
        "⊢ (rain_789 ∨ sunny_12) ∧ cold_333",
        # Proof output
        "exact And.intro h1 h2",
        "exact Or.inl (And.intro h_a h_b)",
        # Full sample
        "state_0:\np1_42 p2_99 : Prop\nh : p1_42\n⊢ p1_42 ∨ p2_99",
    ]

    print("Test tokenization results:")
    for i, text in enumerate(test_cases, 1):
        tokens = sp.encode(text, out_type=str)
        ids = sp.encode(text)
        decoded = sp.decode(ids)

        print(f"\n[Test {i}]")
        print(f"  Input:   {repr(text[:60])}{'...' if len(text) > 60 else ''}")
        print(f"  Tokens:  {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        print(f"  IDs:     {ids[:15]}{'...' if len(ids) > 15 else ''}")
        print(f"  Decoded: {repr(decoded[:60])}{'...' if len(decoded) > 60 else ''}")
        print(f"  Match:   {'✓' if decoded == text else '✗'}")

    # Save config
    config = {
        "corpus_file": corpus_file,
        "vocab_size": vocab_size,
        "model_type": model_type,
        "corpus_lines": line_count,
        "timestamp": datetime.now().isoformat(),
        "model_file": f"{model_prefix}.model",
        "vocab_file": f"{model_prefix}.vocab",
    }

    config_file = output_path / "tokenizer_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print("=" * 70)
    print(f"✅ Tokenizer training complete!")
    print(f"   Model: {model_prefix}.model")
    print(f"   Vocab size: {sp.get_piece_size()}")
    print("=" * 70)

    return f"{model_prefix}.model"


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer on dynamic corpus")
    parser.add_argument(
        "--corpus", default="artifacts/tokenizer/tokenizer_corpus.txt",
        help="Path to corpus file"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/tokenizer/trained",
        help="Output directory"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=2048,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--model-type", choices=['bpe', 'unigram'], default='bpe',
        help="Tokenizer model type"
    )

    args = parser.parse_args()

    train_tokenizer(
        corpus_file=args.corpus,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
