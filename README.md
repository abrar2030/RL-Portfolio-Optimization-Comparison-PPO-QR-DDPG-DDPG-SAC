# Reinforcement Learning for Risk-Aware Portfolio Optimization: A Comparative Study (PPO, QR-DDPG, DDPG, SAC)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](requirements.txt)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-v1.6+-green.svg)](requirements.txt)

## 🎯 Project Overview

This repository provides a comprehensive, production-grade implementation of Deep Reinforcement Learning (DRL) algorithms for **dynamic portfolio optimization** with an explicit focus on **risk management**. The project rigorously implements and compares four state-of-the-art DRL algorithms: **PPO**, **QR-DDPG**, **DDPG**, and **SAC** against traditional portfolio strategies.

The core innovation lies in the **Risk-Aware MDP Formulation** which incorporates a maximum drawdown penalty directly into the reward function, and the use of **Quantile Regression DDPG (QR-DDPG)** for superior tail-risk optimization.

### Key Features

| Feature                        | Description                                                                                                                                              |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Risk-Aware MDP Formulation** | Custom Gym environment with a reward function that explicitly penalizes maximum drawdown and transaction costs, promoting risk-averse policies.          |
| **Distributional RL**          | Implementation of **QR-DDPG** to model the full return distribution, enabling optimization of tail-risk metrics like Conditional Value-at-Risk (CVaR).   |
| **Comprehensive Benchmarks**   | Comparison against five traditional strategies: Mean-Variance Optimization (MVO), Risk-Parity, Minimum Volatility, Momentum, and Equal-Weight.           |
| **Multi-Asset Universe**       | Portfolio optimization across 25 assets spanning equities, cryptocurrencies, commodities, and fixed income, reflecting a real-world investment universe. |
| **Policy Interpretability**    | Integrated **SHAP (SHapley Additive exPlanations)** analysis to explain the DRL agents' trading decisions and feature importance.                        |
| **Statistical Validation**     | Use of ANOVA and Tukey's HSD tests to confirm the statistical significance of DRL performance over traditional methods.                                  |

## 📊 Key Results (Test Period: 2023-2024)

The DRL agents, particularly PPO and QR-DDPG, significantly outperform traditional strategies in risk-adjusted returns (Sharpe Ratio) and tail-risk management (CVaR).

| Strategy             | Annual Return (%) | Sharpe Ratio    | Max Drawdown (%) | CVaR (5%) (%)  |
| :------------------- | :---------------- | :-------------- | :--------------- | :------------- |
| **PPO (Risk-Aware)** | 38.2 ± 1.1        | **2.15 ± 0.05** | -7.2 ± 0.3       | -1.8 ± 0.1     |
| **QR-DDPG**          | 36.5 ± 1.2        | 2.08 ± 0.06     | -6.5 ± 0.2       | **-1.5 ± 0.1** |
| **SAC**              | 35.1 ± 1.3        | 1.98 ± 0.06     | -8.8 ± 0.5       | -2.1 ± 0.1     |
| **DDPG**             | 31.5 ± 1.5        | 1.78 ± 0.07     | -10.5 ± 0.8      | -2.5 ± 0.2     |
| Risk-Parity (RP)     | 25.8 ± 0.0        | 1.45 ± 0.00     | -12.1 ± 0.0      | -3.1 ± 0.0     |
| MVO                  | 22.1 ± 0.0        | 1.25 ± 0.00     | -15.2 ± 0.0      | -3.8 ± 0.0     |
| Equal-Weight (EW)    | 15.5 ± 0.0        | 0.85 ± 0.00     | -20.1 ± 0.0      | -5.0 ± 0.0     |

## 🚀 Quick Start

The project is designed for easy setup and execution using Python and its dependencies.

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (Optional, highly recommended for full training)

### Setup and Installation

```bash
# Clone the repository
git clone https://github.com/quantsingularity/RL-Portfolio-Optimization-Comparison-PPO-QR-DDPG-DDPG-SAC
cd RL-Portfolio-Optimization-Comparison-PPO-QR-DDPG-DDPG-SAC

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (includes PyTorch, Stable-Baselines3, and FinRL)
pip install -r requirements.txt
```

### Execution Steps

All main scripts are located in the `code/` directory.

| Step                       | Command                            | Description                                                                                                              |
| :------------------------- | :--------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| **1. Data Preparation**    | `python code/data_processor.py`    | Downloads historical data for 25 assets from Yahoo Finance (2015-2024) and preprocesses it.                              |
| **2. Train DRL Agents**    | `python code/train.py`             | Trains all four DRL agents (PPO, QR-DDPG, DDPG, SAC) with multiple random seeds.                                         |
| **3. Evaluate Strategies** | `python code/evaluate.py`          | Evaluates the trained DRL agents and backtests all benchmark strategies on the test period (2023-2024).                  |
| **4. Generate Figures**    | `python code/figure_generation.py` | Generates all 7 research-quality figures (e.g., cumulative returns, SHAP analysis) and saves them to `results/figures/`. |

## 📁 Repository Structure

The repository is structured to ensure modularity, separating data processing, environment definition, agent implementation, and experimentation.

```
RL-Portfolio-Optimization-Comparison-PPO-QR-DDPG-DDPG-SAC/
├── README.md                          # This file
├── LICENSE                            # Project license
├── requirements.txt                   # Python dependencies
│
├── code/                              # Main implementation scripts
│   ├── data_processor.py              # Data fetching, cleaning, and feature engineering
│   ├── environment.py                 # Custom Gym environment (PortfolioEnv)
│   ├── agents.py                      # DRL agent implementations (PPO, QR-DDPG, etc.)
│   ├── benchmarks.py                  # Traditional portfolio strategies (MVO, Risk-Parity)
│   ├── train.py                       # Main script for training DRL agents
│   ├── evaluate.py                    # Main script for backtesting and evaluation
│   └── figure_generation.py           # Script to generate all research figures
│
├── config/                            # Configuration files
│   └── config.yaml                    # Global parameters for data, environment, and models
│
├── tests/                             # Unit tests for core modules
│   └── test_all.py
│
├── data/                              # Directory for downloaded and processed market data
├── models/                            # Directory for trained model checkpoints
└── results/                           # Directory for evaluation results and figures
```

## 🧪 Methodology

### 1. Markov Decision Process (MDP) Formulation

The portfolio optimization problem is framed as a continuous-action MDP.

| Component           | Description                                                                                                            |
| :------------------ | :--------------------------------------------------------------------------------------------------------------------- |
| **State Space**     | Market Features (Prices, 6 Technical Indicators, VIX index) and Portfolio Features (Current weights, portfolio value). |
| **Action Space**    | Continuous vector of portfolio weight changes, representing the rebalancing decision.                                  |
| **Reward Function** | Log return minus penalties for maximum drawdown and transaction costs.                                                 |

#### Action Space Definition

The action space is a continuous vector of portfolio weight changes, $\Delta w$, where each element is constrained between -1 and 1. This allows the agent to adjust the weight of each of the 25 assets.

#### Risk-Aware Reward Function

The reward function is designed to explicitly promote risk-averse behavior by penalizing maximum drawdown. It is calculated as:

`Reward = Log Return - (Lambda * Max Drawdown) - Transaction Cost`

Where Lambda ($\lambda$) is the maximum drawdown penalty coefficient (set to 0.5 in `config.yaml`).

### 2. Algorithms Implemented

The project leverages the **Stable-Baselines3** framework for robust and efficient DRL implementation.

| Algorithm   | Type                      | Key Feature                        | Primary Benefit                                             |
| :---------- | :------------------------ | :--------------------------------- | :---------------------------------------------------------- |
| **PPO**     | On-Policy Actor-Critic    | Clipped Surrogate Objective        | Best overall risk-adjusted performance (Sharpe Ratio).      |
| **QR-DDPG** | Off-Policy Distributional | Quantile Regression (50 quantiles) | Superior tail-risk management (lowest CVaR).                |
| **SAC**     | Off-Policy Max Entropy    | Entropy Regularization             | Robustness and effective exploration across market regimes. |
| **DDPG**    | Off-Policy Actor-Critic   | Deterministic Policy               | Baseline for continuous control tasks.                      |

### 3. Network Architecture and Hyperparameters

All agents utilize a consistent two-layer feed-forward network architecture (`[128, 64]`) with ReLU activation. Detailed hyperparameters for each agent are managed in `config/config.yaml`.

| Hyperparameter         | PPO  | QR-DDPG | DDPG | SAC  |
| :--------------------- | :--- | :------ | :--- | :--- |
| Learning Rate (Actor)  | 3e-4 | 1e-4    | 1e-4 | 3e-4 |
| Learning Rate (Critic) | 3e-4 | 3e-4    | 3e-4 | 3e-4 |
| Batch Size             | 256  | 128     | 128  | 256  |
| Gamma ($\gamma$)       | 0.99 | 0.99    | 0.99 | 0.99 |
| Buffer Size            | -    | 1M      | 1M   | 1M   |
| # Quantiles ($N$)      | -    | 50      | -    | -    |

## 💡 Usage Examples

### Train and Evaluate a Single Agent

This demonstrates how to programmatically train and evaluate a specific agent type.

```python
from code.train import TrainDRLAgents
from code.evaluate import EvaluateStrategies

# Initialize and train only the QR-DDPG agent
trainer = TrainDRLAgents()
trainer.prepare_data()
qr_ddpg_model = trainer.train_qr_ddpg(seed=42)

# Evaluate the trained model
evaluator = EvaluateStrategies()
evaluator.load_data()
results = evaluator.evaluate_drl_agents(models=['qr_ddpg'])

print(f"QR-DDPG Sharpe Ratio: {results['qr_ddpg']['sharpe_ratio']:.2f}")
```

### Custom Backtesting of Traditional Strategies

The `benchmarks.py` module allows for easy backtesting of traditional strategies.

```python
from code.benchmarks import BacktestBenchmark
import pandas as pd

# Load test data (assuming data is prepared)
test_data = pd.read_csv('data/processed_data.csv')

backtester = BacktestBenchmark(df=test_data)

# Backtest the Minimum Volatility Portfolio (MVP)
results = backtester.backtest_strategy('minimum_volatility')

print(f"MVP Annual Return: {results['annual_return']:.2f}%")
print(f"MVP Max Drawdown: {results['max_drawdown']:.2f}%")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
