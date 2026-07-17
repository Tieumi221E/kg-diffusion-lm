# KGdLLM: Knowledge Graph Diffusion Language Model Pipeline

KGdLLM is an experimental framework designed to study the capabilities of Discrete Diffusion Language Models (MDM/LLaDA style) in acquiring knowledge and performing logical reasoning on synthetic Knowledge Graph (KG) datasets.

The project features a decoupled architecture, separating the core diffusion algorithms from experiment-specific workflows.

---

## 1. System Architecture

### 1.1 Core Engine (`dllm` package)
The masked-diffusion core (model / forward process / loss / samplers) is provided
by the external [`dllm`](https://github.com/Tieumi221E/dllm) package:
bidirectional Transformer backbone, LLaDA-style forward noising, importance-weighted
masked cross-entropy, and full-canvas / block-wise samplers.

### 1.2 Experiment Scripts (`scripts/`)
Workflow scripts tailored for Knowledge Graph reasoning tasks:
*   **Data Pipeline**: 
    *   `prepare_kg_dataset.py`: Parses raw text into structured Pretrain, SFT, and QA evaluation sets.
*   **Training**:
    *   `train_mdm.py`: Performs bidirectional masked pre-training for knowledge acquisition.
    *   `train_sft.py`: Supervised Fine-Tuning (SFT) for instruction-following and logical reasoning (pure text generation).
*   **Evaluation & Analytics**:
    *   `eval_all_checkpoints.py`: Batch evaluation of model checkpoints across multiple logical dimensions (e.g., reverse relations, multi-hop).
    *   `plot_results.py`: Unified visualization tool for Loss trends and QA accuracy.
    *   `plot_summary.py`: Comparative analysis across different data scales and parameters.

---

## 2. Quick Start

### 2.1 Installation

The code imports `dllm`, the masked-diffusion toolkit maintained in
[`Tieumi221E/dllm`](https://github.com/Tieumi221E/dllm). It is project source
code rather than a PyPI requirement. This release uses commit
`205d08882d2de3305a1b75e2e29613d87e569e5a`.

```bash
git clone https://github.com/Tieumi221E/dllm.git
git -C dllm checkout 205d08882d2de3305a1b75e2e29613d87e569e5a
python -m pip install -e ./dllm
python -m pip install -r requirements.txt
```

### 2.2 Data Preparation
```bash
# Example: Process dataset with scale 1000
python scripts/prepare_kg_dataset.py --dataset-name 1000 --source-subdir txt --tokenizer-config configs/train_mdm.yaml
```

### 2.3 Training & Visualization
```bash
# Run Pre-training
python scripts/train_mdm.py --config configs/train_mdm.yaml --dataset kgd_1000 --output-dir runs/pretrain/kgd_1000

# Run SFT (Requires a pretrained checkpoint)
python scripts/train_sft.py --config configs/sft_mdm.yaml --dataset kgd_1000 --pretrained-checkpoint runs/pretrain/kgd_1000/best.pt --output-dir runs/sft/kgd_1000

# Visualize results
python scripts/plot_results.py --log runs/sft/kgd_1000/train.log --metrics-dir runs/sft/kgd_1000/
```

---

## 3. Directory Structure
*   `configs/`: Experiment configurations (YAML).
*   `datasets/`: Raw and generated intermediate data (ignored by git).
*   `runs/`: Training outputs including checkpoints, logs, and metrics (ignored by git).
*   `history/`: Legacy scripts and archived attempts (ignored by git).

---
*Note: This repository is intended for research purposes. Ensure proper hardware resources (GPU) for training.*

## License

MIT. See [LICENSE](LICENSE).
