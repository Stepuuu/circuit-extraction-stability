<div align="center">

<a href="https://arxiv.org/abs/2607.18921"><img src="assets/readme/header.svg" alt="Circuit Claims Depend on What Is Extracted and How It Is Compared" width="100%"></a>

**Yang Sheng · Jie Fu**

Fudan University · Shanghai Innovation Institute · IQuest Research

[arXiv](https://arxiv.org/abs/2607.18921) &nbsp; · &nbsp; [Read the paper](https://arxiv.org/html/2607.18921v1) &nbsp; · &nbsp; [Citation](#citation)

</div>

## Overview

Code accompanying our paper on circuit extraction in transformers, studied through synthetic Lean tactic-prediction tasks. The repository includes task generation, training, evaluation, and circuit analysis. See the [paper](https://arxiv.org/abs/2607.18921) for the experimental setup and results.

## Installation

Install the dependencies in a Python environment with a PyTorch build compatible with your CUDA version. Training and circuit extraction are intended for a CUDA GPU.

```bash
git clone https://github.com/Stepuuu/circuit-extraction-stability.git
cd circuit-extraction-stability
python -m pip install -r requirements.txt
```

## Using the code

| Directory | Contents |
| :--- | :--- |
| [`data/`](data/) | Synthetic tasks and tokenizer preparation |
| [`training/`](training/) | Supervised training and GRPO |
| [`evaluation/`](evaluation/) | Tactic-prediction evaluation |
| [`circuits/`](circuits/) | Circuit extraction and analysis |
| [`runtime/inference/`](runtime/inference/) | Model definitions and inference utilities |

Start with the tokenizer scripts in [`data/`](data/). Training, evaluation, and extraction options are available through each script's command-line help, for example:

```bash
python training/train_sft.py --help
python evaluation/evaluate.py --help
python circuits/circuit_extraction.py --help
```

The release contains the core scripts. Model checkpoints and generated experiment artifacts are not included; script defaults are not a specification of the full paper experiments.

## Citation

```bibtex
@misc{sheng2026circuitclaimsdependextracted,
  title={Circuit Claims Depend on What Is Extracted and How It Is Compared},
  author={Yang Sheng and Jie Fu},
  year={2026},
  eprint={2607.18921},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2607.18921},
}
```

## License and acknowledgments

Released under the [Apache License 2.0](LICENSE). The inference utilities are adapted from OpenAI's [`circuit_sparsity`](https://github.com/openai/circuit_sparsity); see [NOTICE](NOTICE).
