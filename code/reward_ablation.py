"""
Reward Function Ablation Study Module.

Systematically varies the lambda (drawdown penalty) parameter
to analyze its impact on agent performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import yaml
from pathlib import Path
import torch


class RewardAblationStudy:
    """Perform ablation study on reward function parameters."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ablation study with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.lambda_values = self.config["reward_ablation"]["lambda_values"]
        self.n_seeds = self.config["reward_ablation"]["n_seeds"]
        self.output_dir = Path(self.config["reward_ablation"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_ablation_study(
        self, agent_class, env_factory, training_steps: int = 100000
    ) -> pd.DataFrame:
        """
        Run ablation study across different lambda values.

        Args:
            agent_class: Agent class to train
            env_factory: Function that creates environment with given lambda
            training_steps: Number of training steps per configuration

        Returns:
            DataFrame with results for all configurations
        """
        results = []

        for lambda_val in self.lambda_values:
            print(f"\n{'='*60}")
            print(f"Testing Lambda = {lambda_val}")
            print(f"{'='*60}")

            for seed in range(self.n_seeds):
                print(f"  Seed {seed + 1}/{self.n_seeds}")

                # Set random seeds
                np.random.seed(seed)
                torch.manual_seed(seed)

                # Create environment with specific lambda
                env = env_factory(max_drawdown_penalty=lambda_val)

                # Train agent
                agent = agent_class(env)
                agent.train(total_timesteps=training_steps)

                # Evaluate
                metrics = self._evaluate_agent(agent, env)
                metrics["lambda"] = lambda_val
                metrics["seed"] = seed

                results.append(metrics)

        df = pd.DataFrame(results)

        # Save results
        output_path = self.output_dir / "ablation_results.csv"
        df.to_csv(output_path, index=False)

        return df

    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze ablation study results.

        Args:
            results_df: DataFrame with ablation results

        Returns:
            Dictionary with analysis insights
        """
        analysis = {}

        # Group by lambda
        grouped = results_df.groupby("lambda")

        # Calculate statistics for each lambda
        stats = grouped.agg(
            {
                "sharpe_ratio": ["mean", "std"],
                "annual_return": ["mean", "std"],
                "max_drawdown": ["mean", "std"],
                "cvar": ["mean", "std"],
                "volatility": ["mean", "std"],
            }
        )

        analysis["statistics"] = stats

        # Find optimal lambda for different objectives
        mean_sharpe = grouped["sharpe_ratio"].mean()
        mean_drawdown = grouped["max_drawdown"].mean()
        mean_cvar = grouped["cvar"].mean()

        analysis["optimal_lambda_sharpe"] = mean_sharpe.idxmax()
        analysis["optimal_lambda_drawdown"] = mean_drawdown.idxmax()  # Least negative
        analysis["optimal_lambda_cvar"] = mean_cvar.idxmax()  # Least negative

        # Calculate trade-offs
        analysis["sharpe_vs_drawdown"] = (
            results_df[["lambda", "sharpe_ratio", "max_drawdown"]]
            .groupby("lambda")
            .mean()
        )

        return analysis

    def plot_performance_surface(self, results_df: pd.DataFrame, save_path: str = None):
        """
        Plot 2D/3D performance surface showing lambda impact.

        Args:
            results_df: DataFrame with ablation results
            save_path: Path to save the figure
        """
        fig = plt.figure(figsize=(20, 12))

        # Aggregate results by lambda
        agg_results = results_df.groupby("lambda").agg(
            {
                "sharpe_ratio": ["mean", "std"],
                "annual_return": ["mean", "std"],
                "max_drawdown": ["mean", "std"],
                "cvar": ["mean", "std"],
                "volatility": ["mean", "std"],
            }
        )

        lambda_vals = agg_results.index.values

        # 1. Sharpe Ratio vs Lambda
        ax1 = plt.subplot(2, 3, 1)
        sharpe_mean = agg_results[("sharpe_ratio", "mean")].values
        sharpe_std = agg_results[("sharpe_ratio", "std")].values
        ax1.plot(lambda_vals, sharpe_mean, "o-", linewidth=2, markersize=8)
        ax1.fill_between(
            lambda_vals, sharpe_mean - sharpe_std, sharpe_mean + sharpe_std, alpha=0.3
        )
        ax1.set_xlabel("Lambda (Drawdown Penalty)", fontsize=12)
        ax1.set_ylabel("Sharpe Ratio", fontsize=12)
        ax1.set_title("Sharpe Ratio vs Lambda", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.axvline(
            x=lambda_vals[np.argmax(sharpe_mean)],
            color="red",
            linestyle="--",
            alpha=0.5,
        )

        # 2. Annual Return vs Lambda
        ax2 = plt.subplot(2, 3, 2)
        return_mean = agg_results[("annual_return", "mean")].values
        return_std = agg_results[("annual_return", "std")].values
        ax2.plot(
            lambda_vals, return_mean, "o-", linewidth=2, markersize=8, color="green"
        )
        ax2.fill_between(
            lambda_vals,
            return_mean - return_std,
            return_mean + return_std,
            alpha=0.3,
            color="green",
        )
        ax2.set_xlabel("Lambda (Drawdown Penalty)", fontsize=12)
        ax2.set_ylabel("Annual Return (%)", fontsize=12)
        ax2.set_title("Annual Return vs Lambda", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # 3. Max Drawdown vs Lambda
        ax3 = plt.subplot(2, 3, 3)
        dd_mean = agg_results[("max_drawdown", "mean")].values
        dd_std = agg_results[("max_drawdown", "std")].values
        ax3.plot(lambda_vals, dd_mean, "o-", linewidth=2, markersize=8, color="red")
        ax3.fill_between(
            lambda_vals, dd_mean - dd_std, dd_mean + dd_std, alpha=0.3, color="red"
        )
        ax3.set_xlabel("Lambda (Drawdown Penalty)", fontsize=12)
        ax3.set_ylabel("Max Drawdown (%)", fontsize=12)
        ax3.set_title("Max Drawdown vs Lambda", fontsize=14, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.axvline(
            x=lambda_vals[np.argmax(dd_mean)], color="blue", linestyle="--", alpha=0.5
        )

        # 4. CVaR vs Lambda
        ax4 = plt.subplot(2, 3, 4)
        cvar_mean = agg_results[("cvar", "mean")].values
        cvar_std = agg_results[("cvar", "std")].values
        ax4.plot(
            lambda_vals, cvar_mean, "o-", linewidth=2, markersize=8, color="purple"
        )
        ax4.fill_between(
            lambda_vals,
            cvar_mean - cvar_std,
            cvar_mean + cvar_std,
            alpha=0.3,
            color="purple",
        )
        ax4.set_xlabel("Lambda (Drawdown Penalty)", fontsize=12)
        ax4.set_ylabel("CVaR 5% (%)", fontsize=12)
        ax4.set_title("CVaR vs Lambda", fontsize=14, fontweight="bold")
        ax4.grid(True, alpha=0.3)

        # 5. Volatility vs Lambda
        ax5 = plt.subplot(2, 3, 5)
        vol_mean = agg_results[("volatility", "mean")].values
        vol_std = agg_results[("volatility", "std")].values
        ax5.plot(lambda_vals, vol_mean, "o-", linewidth=2, markersize=8, color="orange")
        ax5.fill_between(
            lambda_vals,
            vol_mean - vol_std,
            vol_mean + vol_std,
            alpha=0.3,
            color="orange",
        )
        ax5.set_xlabel("Lambda (Drawdown Penalty)", fontsize=12)
        ax5.set_ylabel("Volatility (%)", fontsize=12)
        ax5.set_title("Volatility vs Lambda", fontsize=14, fontweight="bold")
        ax5.grid(True, alpha=0.3)

        # 6. Heatmap: Multiple Metrics
        ax6 = plt.subplot(2, 3, 6)
        metrics_matrix = np.array(
            [
                (sharpe_mean - sharpe_mean.min())
                / (sharpe_mean.max() - sharpe_mean.min()),
                (return_mean - return_mean.min())
                / (return_mean.max() - return_mean.min()),
                (dd_mean - dd_mean.min()) / (dd_mean.max() - dd_mean.min()),
            ]
        )

        sns.heatmap(
            metrics_matrix,
            xticklabels=[f"{l:.1f}" for l in lambda_vals],
            yticklabels=["Sharpe", "Return", "Drawdown"],
            cmap="RdYlGn",
            annot=False,
            fmt=".2f",
            ax=ax6,
            cbar_kws={"label": "Normalized Performance"},
        )
        ax6.set_xlabel("Lambda", fontsize=12)
        ax6.set_title(
            "Performance Heatmap (Normalized)", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_tradeoff_frontier(self, results_df: pd.DataFrame, save_path: str = None):
        """
        Plot return vs risk trade-off frontier for different lambda values.

        Args:
            results_df: DataFrame with ablation results
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Aggregate by lambda
        agg = results_df.groupby("lambda").agg(
            {
                "annual_return": "mean",
                "volatility": "mean",
                "max_drawdown": "mean",
                "sharpe_ratio": "mean",
            }
        )

        # Create scatter plot
        scatter = ax.scatter(
            agg["volatility"],
            agg["annual_return"],
            c=agg.index,
            s=200,
            cmap="viridis",
            edgecolors="black",
            linewidths=2,
            alpha=0.7,
        )

        # Annotate points with lambda values
        for idx, row in agg.iterrows():
            ax.annotate(
                f"λ={idx:.1f}",
                (row["volatility"], row["annual_return"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Lambda (Drawdown Penalty)", fontsize=12)

        ax.set_xlabel("Volatility (%)", fontsize=14)
        ax.set_ylabel("Annual Return (%)", fontsize=14)
        ax.set_title(
            "Risk-Return Trade-off Frontier\n(Impact of Lambda Parameter)",
            fontsize=16,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def generate_ablation_report(
        self, results_df: pd.DataFrame, analysis: Dict, output_path: str = None
    ) -> str:
        """Generate comprehensive ablation study report."""
        report = []
        report.append("=" * 80)
        report.append("REWARD FUNCTION ABLATION STUDY REPORT")
        report.append("=" * 80)
        report.append("")

        # Study configuration
        report.append("STUDY CONFIGURATION")
        report.append("-" * 80)
        report.append(f"Lambda values tested: {self.lambda_values}")
        report.append(f"Number of seeds per configuration: {self.n_seeds}")
        report.append(f"Total experiments: {len(results_df)}")
        report.append("")

        # Optimal lambda values
        report.append("OPTIMAL LAMBDA VALUES")
        report.append("-" * 80)
        report.append(
            f"Best for Sharpe Ratio: λ = {analysis['optimal_lambda_sharpe']:.2f}"
        )
        report.append(
            f"Best for Max Drawdown: λ = {analysis['optimal_lambda_drawdown']:.2f}"
        )
        report.append(f"Best for CVaR: λ = {analysis['optimal_lambda_cvar']:.2f}")
        report.append("")

        # Performance statistics
        report.append("PERFORMANCE STATISTICS BY LAMBDA")
        report.append("-" * 80)
        report.append(analysis["statistics"].to_string())
        report.append("")

        # Key insights
        report.append("KEY INSIGHTS")
        report.append("-" * 80)

        sharpe_df = results_df.groupby("lambda")["sharpe_ratio"].mean()
        best_lambda = sharpe_df.idxmax()
        worst_lambda = sharpe_df.idxmin()

        report.append(f"1. Sharpe ratio peaks at λ = {best_lambda:.2f}")
        report.append(
            f"2. Performance degradation at extreme values (λ = {worst_lambda:.2f})"
        )
        report.append(
            f"3. Recommended range: λ ∈ [{max(0, best_lambda - 0.2):.1f}, {min(1.0, best_lambda + 0.2):.1f}]"
        )
        report.append("")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)

        return report_text

    def _evaluate_agent(self, agent, env) -> Dict:
        """Evaluate trained agent and return metrics."""
        # Run evaluation episodes
        n_episodes = 10
        episode_returns = []
        episode_sharpes = []
        episode_drawdowns = []

        for _ in range(n_episodes):
            obs = env.reset()
            done = False

            while not done:
                action = agent.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

            metrics = env.get_portfolio_metrics()
            episode_returns.append(metrics.get("annual_return", 0) * 100)
            episode_sharpes.append(metrics.get("sharpe_ratio", 0))
            episode_drawdowns.append(metrics.get("max_drawdown", 0))

        return {
            "annual_return": np.mean(episode_returns),
            "sharpe_ratio": np.mean(episode_sharpes),
            "max_drawdown": np.mean(episode_drawdowns),
            "cvar": np.mean(
                [
                    env.get_portfolio_metrics().get("cvar_5", 0)
                    for _ in range(n_episodes)
                ]
            ),
            "volatility": np.mean(
                [
                    env.get_portfolio_metrics().get("volatility", 0)
                    for _ in range(n_episodes)
                ]
            )
            * 100,
        }


if __name__ == "__main__":
    print("Reward Ablation Study module loaded successfully")
