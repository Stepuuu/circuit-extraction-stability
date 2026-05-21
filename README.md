# Supplementary Code

This repository contains the core scripts used for the controlled circuit-reproducibility study.

## Structure

- `data/`: synthetic logic-task generation and tokenizer training.
- `training/`: supervised pretraining and GRPO fine-tuning scripts.
- `evaluation/`: task-level evaluation scripts.
- `circuits/`: circuit extraction, factorized graph extraction, and fixed-loss pruning sweeps.
- `runtime/inference/`: minimal model and hook utilities required by the scripts above.

Large artifacts such as checkpoints, generated data, logs, and result tables are intentionally not included.

The `runtime/inference/` utilities are adapted from OpenAI's Apache-2.0 [`circuit_sparsity`](https://github.com/openai/circuit_sparsity) codebase, with local adaptations for this study; see `LICENSE` and `NOTICE`.

## Environment

```bash
pip install -r requirements.txt
```

## Typical Workflow

```bash
python data/generate_tokenizer_corpus.py
python data/train_tokenizer.py
python training/train_sft.py --config dense
python training/train_grpo.py --model-type custom --model-path <checkpoint> --config-path <config> --output-dir artifacts/results/grpo_run
python evaluation/evaluate.py --results-dir artifacts/results/checkpoints
python circuits/circuit_extraction.py --checkpoint <checkpoint>/best_model.pt --config <checkpoint>/config.json --output-dir artifacts/results/circuits --task-group composition
```

The factorized query/key and fixed-loss pruning scripts in `circuits/` use the same checkpoint/config/output-dir convention.

## License

Apache License 2.0 (see `LICENSE`).
