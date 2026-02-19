# SparseMultiTaskBNN

Sparse Multi-Task Bayesian Neural Networks with Cluster-Based Active Learning.

This repository contains the source code for the paper:

> Xiong, F., **Wang, B.**, Li, C., Yin, J., & Wang, H. (2025). A comprehensive multi-source uncertainty quantification method for RANS-CFD based on sparse multi-task deep active learning. *Aerospace Science and Technology*, 165, 110483. https://doi.org/10.1016/j.ast.2025.110483

## Overview

The framework combines three key ideas for data-efficient multi-task regression with uncertainty quantification:

1. **Variational Bayesian Last Layer (VBLL)** — The final layer of each task head is a variational Bayesian linear layer, enabling principled epistemic and aleatoric uncertainty estimation without the cost of full-network BNN inference.

2. **Sparse Subnetwork Discovery via Iterative Magnitude Pruning** — An overparameterized shared backbone is pruned per-task to discover task-specific sparse subnetworks (Lottery Ticket Hypothesis style). Different tasks share overlapping but distinct subsets of the backbone weights.

3. **Cluster-Based Active Learning with Expected Model Change** — Candidate points are clustered by predictive uncertainty, and within each cluster the most informative sample is selected via Expected Gradient Length (EGL), with dynamic distance thresholds to prevent redundancy.

Training uses **PCGrad** (Projecting Conflicting Gradients) to resolve gradient conflicts across tasks on the shared backbone.

## Architecture

```
                         Input
                           |
                    +--------------+
                    |   BaseMlp    |  (shared overparameterized backbone)
                    |  per-task    |
                    |  binary mask |
                    +--------------+
                     /     |      \
              +-------+ +-------+ +-------+
              |TaskMlp| |TaskMlp| |TaskMlp|  (task-specific branches)
              | +VBLL | | +VBLL | | +VBLL |  (Bayesian last layer)
              +-------+ +-------+ +-------+
                 |          |         |
              Task 1     Task 2    Task 3
            (mean, var) (mean, var) (mean, var)
```

## Project Structure

```
SparseMultiTaskBNN/
├── models/                          # Model definitions
│   ├── distributions.py             # Gaussian parameterizations (diagonal, dense, low-rank, precision)
│   ├── regression.py                # VBLL regression (Gaussian & Student-t)
│   ├── multi_vbll.py                # Multi-task network with sparse masks
│   └── decorator.py                 # Thread-safe metadata decorator
├── trainer/                         # Training utilities
│   ├── multi_trainer.py             # PCGrad & random task sampling trainers
│   ├── gradient_surgery.py          # PCGrad implementation
│   ├── sparse.py                    # Iterative magnitude pruning
│   └── weighted_random.py           # Weighted task sampling
├── active_learning/                 # Active learning strategies
│   ├── active_learning_cluster.py   # Cluster-based AL with EGL acquisition
│   ├── cluster_sigma.py             # Cluster-based sigma/EGL acquisition
│   ├── directly_sigma.py            # Direct uncertainty-based selection
│   ├── expected_model_change.py     # Expected Gradient Length computation
│   ├── dynamic_threshold.py         # Adaptive distance threshold
│   ├── euclidean_distance.py        # Pairwise distance utilities
│   └── inverse_distance.py          # Inverse distance weighting
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/bominwang/SparseMultiTaskBNN.git
cd SparseMultiTaskBNN
pip install -r requirements.txt
```

## Quick Start

```python
from models.multi_vbll import MultiTaskVBLL
from trainer.multi_trainer import PcGrad_Trainer
from trainer.sparse import get_sparse_network

# Define model configuration
config_base = {
    'input_features': 3,
    'hidden_features': 64,
    'num_hidden': 3
}

config_task = {
    'num_task': 2,
    'input_features': [0, 0],       # extra per-task features (0 = none)
    'output_features': [1, 1],
    'hidden_features': 32,
    'num_hidden': 2,
    'prior_scale': 1.0,
    'reg_weight': 1e-4,
    'wishart_scale': 1e-2,
    'parameterization': 'diagonal',
    'dof': 1.0
}

# Build model
model = MultiTaskVBLL(config_base, config_task)

# Train with PCGrad
model = PcGrad_Trainer(model, multi_task_loaders, lr=1e-3, epochs=200, device='cpu')

# Discover sparse subnetworks
model = get_sparse_network(model, train_loaders, test_loaders,
                           prune_ratio=0.3, prune_epochs=5,
                           prune_train_epochs=50, prune_learn_rate=1e-3,
                           device='cpu')

# Predict with uncertainty
output = model(task_id=0, x=test_input)
mean = output.predictive.mean       # predictive mean
var = output.predictive.variance    # predictive variance (epistemic + aleatoric)
```

## Citation

If you find this code useful in your research, please cite:

```bibtex
@article{xiong2025comprehensive,
  title={A comprehensive multi-source uncertainty quantification method for RANS-CFD based on sparse multi-task deep active learning},
  author={Xiong, FenFen and Wang, Bomin and Li, Chao and Yin, Jianhua and Wang, Haoyu},
  journal={Aerospace Science and Technology},
  volume={165},
  pages={110483},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.ast.2025.110483}
}
```
