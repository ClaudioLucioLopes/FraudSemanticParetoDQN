# FraudSemanticParetoDQN

A Multi-Objective Reinforcement Learning (MORL) system for fraud detection using Pareto-DQN with semantic embeddings. This project implements a sophisticated agent capable of balancing multiple objectives (e.g., Efficiency, Drift, and Diversity) to identify fraudulent transactions in a multi-objective environment.

## Overview

The core of this repository is a **Pareto-DQN** agent that approximates the Pareto frontier of non-dominated solutions. Unlike standard DQN, which collapses multiple objectives into a single scalar reward, this agent maintains the vector-valued nature of the problem, allowing for more flexible decision-making (e.g., via Compromise Programming).

### Key Components

- **`ParetoDQN.py`**: Implementation of the Hypervolume-based Pareto agent and its neural network approximators.
- **`FraudMOEnv.py`**: A Multi-Objective Environment for fraud detection scenarios.
- **`FraudDataHandlerImproved.py`**: Data preprocessing pipeline featuring semantic embedding generation (using `sentence-transformers`).
- **`StandardDQN.py`**: A baseline scalar DQN implementation for performance comparison.
- **`FraudBaselineXGB.py`**: A baseline XGBoost classifier for benchmarking.

## Architecture

The system uses a **Double Pareto-DQN** architecture:
1. **Reward Approximator**: Predicts the immediate vectorial reward $R(s, a)$.
2. **Non-Dominated Approximator**: Approximates the Pareto frontier surface, predicting future returns.
3. **Hypervolume Selection**: During training, actions are selected to maximize the topological hypervolume.
4. **Compromise Programming**: During inference, the agent uses the **Augmented Tchebycheff Distance** to the Utopia Point to select the mathematically optimal trade-off.

## Installation

```bash
pip install -r requirements.txt
```

*Note: This project requires PyTorch and several scientific libraries (pymoo, scikit-learn, sentence-transformers, xgboost).*

## Usage

### Training the Pareto-DQN Agent
```bash
python FraudParetoDQN.py
```

### Running Baselines
```bash
python FraudStandardDQN.py
python FraudBaselineXGB.py
```

### Generating Visualizations
```bash
python plot_roc_pr_curves.py
```

## Repository Structure

```text
├── all-MiniLM-L6-v2/     # Local sentence-transformer model (ignored)
├── data/                 # Raw and processed datasets (ignored)
├── textos/               # Reference documents and PDFs (ignored)
├── ParetoDQN.py          # Core Pareto-DQN logic
├── FraudMOEnv.py         # Multi-objective fraud environment
├── FraudDataHandler.py   # Semantic embedding & data handling
└── results/              # Output metrics and plots (ignored)
```

## License

This project is developed for research purposes in the field of Multi-Objective Reinforcement Learning and Fraud Detection.
