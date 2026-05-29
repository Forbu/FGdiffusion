# FGdiffusion

**Large-Scale Knowledge Graph Generation Using a Diffusion Approach**

[![Paper](https://img.shields.io/badge/Paper-CEUR_WS-blue)](https://ceur-ws.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code for the paper *"FGdiffusion: Large-Scale Knowledge Graph Generation Using a Diffusion Approach"* (Adrien Bufort, Lionel Tailhardat, 2025).

## Overview

FGdiffusion is a framework for generating graphs — including knowledge graphs — using discrete diffusion models. It implements and compares several generative approaches for graph generation, with a focus on **discrete diffusion** applied to edge-level graph tokens.

### Key Contributions

- **Discrete diffusion for graph generation**: We adapt LLaDA-style discrete diffusion and score-based discrete diffusion (SEDD) to the graph generation task, operating directly on edge token sequences.
- **Graph flattening via BFS ordering**: Graphs are serialized into edge-index sequences using BFS node ordering, enabling transformer-based models to process graphs as token sequences.
- **Knowledge graph generation**: We extend the approach to generate typed knowledge graphs with node and edge labels.

## Implemented Models

| Model | Type | File | Reference |
|-------|------|------|-----------|
| **G2PT (Autoregressive)** | Autoregressive Transformer | `deepgraphgen/trainers/trainer_g2pt_auto.py` | [Mao et al., 2025](https://arxiv.org/abs/2501.01073) |
| **G2PT + LLaDA** | Discrete diffusion | `deepgraphgen/trainers/trainer_g2pt_llada.py` | [Xie et al., 2025](https://arxiv.org/abs/2502.09992) |
| **G2PT + Score Diffusion** | Score-based discrete diffusion | `deepgraphgen/trainers/trainer_g2pt_score.py` | Based on [SEDD](https://arxiv.org/abs/2310.16834) 
| **G2PT + KG (NASA)** | KG generation with labels | `deepgraphgen/trainers/trainer_g2pt_llada_kg.py` | This paper |

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/FGdiffusion.git
cd FGdiffusion

# Install dependencies
pip install torch networkx torch_geometric lightning x-transformers matplotlib pytest heavyball tensorboardX einops
```

### Docker

```bash
docker build -t fgdiffusion .
docker run --gpus all -it fgdiffusion
```

## Quick Start

### Training

```bash
# Train G2PT with LLaDA discrete diffusion on planar graphs
python scripts/train_g2pt_llada.py

# Train G2PT with score-based diffusion
python scripts/train_g2pt_score.py

# Train GraphGDP baseline
python scripts/train_diffusion.py

# Train GRAN baseline
python scripts/train_gran.py
```

### Evaluation

```bash
# Evaluate generated graphs
python scripts/evaluate.py --checkpoint path/to/checkpoint.ckpt
```

### Knowledge Graph Generation

```bash
# Train on NASA Knowledge Graph
python scripts/train_kg.py
```

## Graph Representation

Graphs are flattened into sequences of edge tokens using the following approach:

1. **BFS ordering**: Nodes are ordered via BFS traversal starting from node 0
2. **Edge serialization**: Edges are serialized as pairs of node indices `(u, v)`
3. **Padding**: Edge sequences are padded to a fixed length based on the `edges_to_node_ratio`
4. **Mask tokens**: A special `[MASK]` token (index `nb_max_node + 2`) is used for discrete diffusion

## Results

Results on synthetic graph datasets (MMD metrics — lower is better):

| Model | Dataset | MMD Degree | MMD Clustering | MMD Orbits |
|-------|---------|------------|----------------|------------|
| G2PT + LLaDA | Planar | 0.0955 | 0.0946 | 0.2298 |
| G2PT + LLaDA | Tree | 0.0138 | 0.0059 | 0.1819 |
| G2PT + SEDD | Tree | 0.0183 | 0.0000 | 0.2641 |

### Key Findings

- **Graph Transformers outperform GNNs** for graph generation, especially in diffusion setups
- **Flattening to edge indices** works better than adjacency matrix representations
- **Discrete diffusion** (LLaDA-like) is effective for graph generation
- **BFS node ordering** significantly enhances learning

## Project Structure

```
FGdiffusion/
├── deepgraphgen/
│   ├── trainers/                # Training modules (PyTorch Lightning)
│   │   ├── trainer_g2pt_auto.py # Autoregressive G2PT
│   │   ├── trainer_g2pt_llada.py # G2PT + LLaDA discrete diffusion
│   │   ├── trainer_g2pt_score.py # G2PT + score-based diffusion
│   │   └── trainer_g2pt_llada_kg.py # G2PT + KG with labels
│   ├── datageneration.py        # Graph data generation utilities
│   ├── datasets.py              # Dataset classes for all approaches
│   ├── diffusion_generation.py  # Diffusion noise scheduling
│   ├── utils.py                 # Shared utilities
│   └── random_walk_features.py  # Random walk feature computation
├── scripts/                     # Training & evaluation scripts
├── tests/                       # Unit tests
├── scripts_preprocess/          # Data preprocessing notebooks
├── images/                      # Example generation images
├── data_kg/                     # Knowledge graph data (parquet)
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{bufort2025fgdiffusion,
  title={FGdiffusion: Large-Scale Knowledge Graph Generation Using a Diffusion Approach},
  author={Bufort, Adrien and Tailhardat, Lionel},
  booktitle={CEUR Workshop Proceedings},
  year={2025}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
