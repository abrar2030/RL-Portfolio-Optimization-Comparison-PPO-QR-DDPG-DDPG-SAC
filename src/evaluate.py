"""
Evaluation script for comparing all strategies (DRL + Benchmarks).

This script:
1. Evaluates trained DRL agents
2. Backtests benchmark strategies
3. Performs statistical significance testing
4. Generates comparison tables and figures
"""

import os
import sys
import yaml
import pandas as pd
import torch
from stable_baselines3 import PPO, DDPG, SAC
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from environment import PortfolioEnv
from agents import QRDDPGAgent
from benchmarks import BacktestBenchmark


class EvaluateStrategies:
    """Evaluate and compare all portfolio strategies."""

    def __init__(self, config_path: str = "../config/config.yaml"):
        """Initialize evaluator."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.results_dir = self.config["output"]["results_dir"]
        self.models_dir = self.config["output"]["models_dir"]

        os.makedirs(self.results_dir, exist_ok=True)

        print("Evaluation configuration loaded")

    def load_data(self):
        """Load and process data."""
        processor = DataProcessor(self.config)
        _, self.test_data = processor.process_all()

        print(f"Test data loaded: {self.test_data.shape}")

    def evaluate_drl_agents(self):
        """Evaluate all trained DRL agents."""
        print("\n" + "=" * 50)
        print("Evaluating DRL Agents")
        print("=" * 50)

        results = []
        n_seeds = self.config["training"]["n_seeds"]

        # Evaluate PPO
        print("\nEvaluating PPO...")
        for seed in range(n_seeds):
            model_path = os.path.join(self.models_dir, f"ppo_seed_{seed}")
            if os.path.exists(model_path + ".zip"):
                model = PPO.load(model_path)
                metrics, portfolio_df = self._evaluate_sb3_agent(model, "PPO", seed)
                results.append(metrics)

                # Save portfolio values for first seed
                if seed == 0:
                    portfolio_df.to_csv(
                        os.path.join(self.results_dir, "ppo_portfolio_values.csv"),
                        index=False,
                    )

        # Evaluate DDPG
        print("\nEvaluating DDPG...")
        for seed in range(n_seeds):
            model_path = os.path.join(self.models_dir, f"ddpg_seed_{seed}")
            if os.path.exists(model_path + ".zip"):
                model = DDPG.load(model_path)
                metrics, portfolio_df = self._evaluate_sb3_agent(model, "DDPG", seed)
                results.append(metrics)

                if seed == 0:
                    portfolio_df.to_csv(
                        os.path.join(self.results_dir, "ddpg_portfolio_values.csv"),
                        index=False,
                    )

        # Evaluate SAC
        print("\nEvaluating SAC...")
        for seed in range(n_seeds):
            model_path = os.path.join(self.models_dir, f"sac_seed_{seed}")
            if os.path.exists(model_path + ".zip"):
                model = SAC.load(model_path)
                metrics, portfolio_df = self._evaluate_sb3_agent(model, "SAC", seed)
                results.append(metrics)

                if seed == 0:
                    portfolio_df.to_csv(
                        os.path.join(self.results_dir, "sac_portfolio_values.csv"),
                        index=False,
                    )

        # Evaluate QR-DDPG
        print("\nEvaluating QR-DDPG...")
        for seed in range(n_seeds):
            model_path = os.path.join(self.models_dir, f"qr_ddpg_seed_{seed}.pt")
            if os.path.exists(model_path):
                metrics, portfolio_df = self._evaluate_qr_ddpg_agent(model_path, seed)
                results.append(metrics)

                if seed == 0:
                    portfolio_df.to_csv(
                        os.path.join(self.results_dir, "qr_ddpg_portfolio_values.csv"),
                        index=False,
                    )

        # Save DRL results
        drl_df = pd.DataFrame(results)
        drl_df.to_csv(
            os.path.join(self.results_dir, "drl_evaluation_results.csv"), index=False
        )

        return drl_df

    def _evaluate_sb3_agent(self, model, agent_name: str, seed: int):
        """Evaluate Stable-Baselines3 agent."""
        env = PortfolioEnv(
            df=self.test_data,
            initial_amount=self.config["environment"]["initial_amount"],
            transaction_cost_pct=self.config["environment"]["transaction_cost_pct"],
            max_drawdown_penalty=self.config["risk"]["max_drawdown_penalty"],
            hmax=self.config["environment"]["hmax"],
            print_verbosity=1000,
        )

        state = env.reset()
        done = False

        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, info = env.step(action)

        metrics = env.get_portfolio_metrics()
        metrics["agent"] = agent_name
        metrics["seed"] = seed

        portfolio_df = env.save_portfolio_values()

        return metrics, portfolio_df

    def _evaluate_qr_ddpg_agent(self, model_path: str, seed: int):
        """Evaluate QR-DDPG agent."""
        env = PortfolioEnv(
            df=self.test_data,
            initial_amount=self.config["environment"]["initial_amount"],
            transaction_cost_pct=self.config["environment"]["transaction_cost_pct"],
            max_drawdown_penalty=self.config["risk"]["max_drawdown_penalty"],
            hmax=self.config["environment"]["hmax"],
            print_verbosity=1000,
        )

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = QRDDPGAgent(state_dim=state_dim, action_dim=action_dim, device=device)

        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])

        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, noise=0.0)
            state, reward, done, info = env.step(action)

        metrics = env.get_portfolio_metrics()
        metrics["agent"] = "QR-DDPG"
        metrics["seed"] = seed

        portfolio_df = env.save_portfolio_values()

        return metrics, portfolio_df

    def evaluate_benchmarks(self):
        """Evaluate benchmark strategies."""
        print("\n" + "=" * 50)
        print("Evaluating Benchmark Strategies")
        print("=" * 50)

        backtester = BacktestBenchmark(
            df=self.test_data,
            initial_amount=self.config["environment"]["initial_amount"],
            transaction_cost_pct=self.config["environment"]["transaction_cost_pct"],
            rebalance_freq=20,
        )

        results = []

        strategies = [
            "equal_weight",
            "mvo",
            "risk_parity",
            "minimum_volatility",
            "momentum",
        ]

        for strategy in strategies:
            print(f"\nBacktesting {strategy}...")
            result = backtester.backtest_strategy(strategy)

            # Format results
            metrics = {
                "agent": strategy.upper(),
                "seed": 0,
                "annual_return": result["annual_return"],
                "sharpe_ratio": result["sharpe_ratio"],
                "sortino_ratio": result["sortino_ratio"],
                "max_drawdown": result["max_drawdown"],
                "cvar_5": result["cvar_5"],
                "volatility": result["volatility"],
            }

            results.append(metrics)

            # Save portfolio values
            portfolio_df = pd.DataFrame(
                {"date": result["dates"], "portfolio_value": result["portfolio_values"]}
            )

            portfolio_df.to_csv(
                os.path.join(self.results_dir, f"{strategy}_portfolio_values.csv"),
                index=False,
            )

        benchmark_df = pd.DataFrame(results)
        benchmark_df.to_csv(
            os.path.join(self.results_dir, "benchmark_evaluation_results.csv"),
            index=False,
        )

        return benchmark_df

    def statistical_significance_test(
        self, drl_df: pd.DataFrame, benchmark_df: pd.DataFrame
    ):
        """Perform statistical significance testing."""
        print("\n" + "=" * 50)
        print("Statistical Significance Testing")
        print("=" * 50)

        # For simplicity, use annual returns for testing
        # In practice, you'd use daily returns

        # Combine results
        all_results = pd.concat([drl_df, benchmark_df], ignore_index=True)

        # Group by agent
        groups = [
            group["annual_return"].values
            for name, group in all_results.groupby("agent")
        ]
        list(all_results.groupby("agent").groups.keys())

        # ANOVA test
        f_stat, p_value = f_oneway(*groups)

        print(f"\nANOVA Test:")
        print(f"F-statistic: {f_stat:.2f}")
        print(f"P-value: {p_value:.6f}")

        if p_value < 0.05:
            print("Result: Significant difference between strategies (p < 0.05)")
        else:
            print("Result: No significant difference between strategies (p >= 0.05)")

        # Tukey's HSD test (pairwise comparisons)
        print("\nTukey's HSD Test:")

        # Prepare data for Tukey test
        tukey_data = []
        tukey_labels = []

        for name, group in all_results.groupby("agent"):
            tukey_data.extend(group["annual_return"].values)
            tukey_labels.extend([name] * len(group))

        tukey_result = pairwise_tukeyhsd(tukey_data, tukey_labels, alpha=0.05)
        print(tukey_result)

        # Save results
        tukey_df = pd.DataFrame(
            data=tukey_result.summary().data[1:], columns=tukey_result.summary().data[0]
        )
        tukey_df.to_csv(
            os.path.join(self.results_dir, "tukey_hsd_results.csv"), index=False
        )

    def create_comparison_table(self, drl_df: pd.DataFrame, benchmark_df: pd.DataFrame):
        """Create comprehensive comparison table."""
        print("\n" + "=" * 50)
        print("Creating Comparison Table")
        print("=" * 50)

        # Combine results
        all_results = pd.concat([drl_df, benchmark_df], ignore_index=True)

        # Calculate mean and std for each agent
        summary = (
            all_results.groupby("agent")
            .agg(
                {
                    "annual_return": ["mean", "std"],
                    "sharpe_ratio": ["mean", "std"],
                    "sortino_ratio": ["mean", "std"],
                    "max_drawdown": ["mean", "std"],
                    "cvar_5": ["mean", "std"],
                }
            )
            .round(2)
        )

        # Format as "mean ± std"
        formatted_results = []

        for agent in summary.index:
            row = {"Strategy": agent}

            for metric in [
                "annual_return",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown",
                "cvar_5",
            ]:
                mean_val = summary.loc[agent, (metric, "mean")]
                std_val = summary.loc[agent, (metric, "std")]
                row[metric] = f"{mean_val:.2f} ± {std_val:.2f}"

            formatted_results.append(row)

        comparison_df = pd.DataFrame(formatted_results)

        # Rename columns
        comparison_df.columns = [
            "Strategy",
            "Annual Return (%)",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown (%)",
            "CVaR (5%) (%)",
        ]

        # Save table
        comparison_df.to_csv(
            os.path.join(self.results_dir, "comparison_table.csv"), index=False
        )

        print("\nComparison Table:")
        print(comparison_df.to_string(index=False))

        return comparison_df

    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        # Load data
        self.load_data()

        # Evaluate DRL agents
        drl_df = self.evaluate_drl_agents()

        # Evaluate benchmarks
        benchmark_df = self.evaluate_benchmarks()

        # Statistical testing
        self.statistical_significance_test(drl_df, benchmark_df)

        # Create comparison table
        comparison_df = self.create_comparison_table(drl_df, benchmark_df)

        print("\n" + "=" * 50)
        print("Evaluation completed successfully!")
        print("=" * 50)

        return drl_df, benchmark_df, comparison_df


def main():
    """Main evaluation function."""
    evaluator = EvaluateStrategies()
    drl_df, benchmark_df, comparison_df = evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
