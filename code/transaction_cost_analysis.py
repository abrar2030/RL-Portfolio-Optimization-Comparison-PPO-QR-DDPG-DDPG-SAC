"""
Transaction Cost Analysis Module.

Analyzes portfolio performance under different transaction cost structures
and rebalancing frequencies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import yaml
from pathlib import Path


class TransactionCostAnalyzer:
    """Analyze impact of transaction costs on portfolio performance."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize analyzer with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.cost_structures = self.config["transaction_costs"]["cost_structures"]
        self.rebalance_frequencies = self.config["transaction_costs"][
            "rebalance_frequencies"
        ]
        self.output_dir = Path(self.config["transaction_costs"]["analysis_output"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_strategy_with_costs(
        self,
        strategy_name: str,
        portfolio_values_no_cost: List[float],
        portfolio_weights_history: List[np.ndarray],
        dates: List,
        cost_structure: str = "standard",
        rebalance_freq: int = 20,
    ) -> Dict:
        """
        Analyze strategy performance with transaction costs.

        Args:
            strategy_name: Name of the strategy
            portfolio_values_no_cost: Portfolio values without transaction costs
            portfolio_weights_history: History of portfolio weights
            dates: List of dates
            cost_structure: Type of cost structure to apply
            rebalance_freq: Rebalancing frequency in days

        Returns:
            Dictionary with performance metrics
        """
        cost_pct = self.cost_structures[cost_structure]

        # Calculate portfolio values with transaction costs
        portfolio_values_with_cost = [portfolio_values_no_cost[0]]
        total_transaction_costs = 0

        for i in range(1, len(portfolio_values_no_cost)):
            # Check if rebalancing occurs
            if i % rebalance_freq == 0 and i > 0:
                # Calculate weight changes
                prev_weights = portfolio_weights_history[i - 1]
                curr_weights = portfolio_weights_history[i]
                weight_changes = np.abs(curr_weights - prev_weights)

                # Calculate transaction cost
                transaction_cost = (
                    np.sum(weight_changes) * portfolio_values_no_cost[i - 1] * cost_pct
                )
                total_transaction_costs += transaction_cost

                # Apply cost
                new_value = portfolio_values_no_cost[i] - transaction_cost
            else:
                new_value = portfolio_values_no_cost[i]

            portfolio_values_with_cost.append(new_value)

        # Calculate metrics
        returns_no_cost = (
            np.diff(portfolio_values_no_cost) / portfolio_values_no_cost[:-1]
        )
        returns_with_cost = (
            np.diff(portfolio_values_with_cost) / portfolio_values_with_cost[:-1]
        )

        metrics = {
            "strategy": strategy_name,
            "cost_structure": cost_structure,
            "rebalance_freq": rebalance_freq,
            "total_transaction_costs": total_transaction_costs,
            "final_value_no_cost": portfolio_values_no_cost[-1],
            "final_value_with_cost": portfolio_values_with_cost[-1],
            "cost_impact": portfolio_values_no_cost[-1]
            - portfolio_values_with_cost[-1],
            "cost_impact_pct": (
                portfolio_values_no_cost[-1] - portfolio_values_with_cost[-1]
            )
            / portfolio_values_no_cost[-1]
            * 100,
            "sharpe_no_cost": self._calculate_sharpe(returns_no_cost),
            "sharpe_with_cost": self._calculate_sharpe(returns_with_cost),
            "max_drawdown_no_cost": self._calculate_max_drawdown(
                portfolio_values_no_cost
            ),
            "max_drawdown_with_cost": self._calculate_max_drawdown(
                portfolio_values_with_cost
            ),
        }

        return metrics

    def analyze_rebalancing_frequency(
        self,
        strategy_name: str,
        portfolio_values_base: List[float],
        portfolio_weights_history: List[np.ndarray],
        dates: List,
    ) -> pd.DataFrame:
        """
        Analyze optimal rebalancing frequency considering transaction costs.

        Returns:
            DataFrame with performance metrics for each frequency
        """
        results = []

        for freq_name, freq_days in self.rebalance_frequencies.items():
            for cost_name in self.cost_structures.keys():
                metrics = self.analyze_strategy_with_costs(
                    strategy_name,
                    portfolio_values_base,
                    portfolio_weights_history,
                    dates,
                    cost_structure=cost_name,
                    rebalance_freq=freq_days,
                )
                metrics["frequency_name"] = freq_name
                results.append(metrics)

        return pd.DataFrame(results)

    def compare_with_without_costs(
        self, strategies: Dict[str, Dict], output_path: str = None
    ) -> pd.DataFrame:
        """
        Compare all strategies with and without transaction costs.

        Args:
            strategies: Dictionary of strategy results
            output_path: Path to save comparison table

        Returns:
            Comparison DataFrame
        """
        comparison_data = []

        for strategy_name, strategy_data in strategies.items():
            # Without costs
            comparison_data.append(
                {
                    "Strategy": strategy_name,
                    "Scenario": "No Costs",
                    "Final Value": strategy_data["final_value_no_cost"],
                    "Annual Return (%)": strategy_data["annual_return_no_cost"],
                    "Sharpe Ratio": strategy_data["sharpe_no_cost"],
                    "Max Drawdown (%)": strategy_data["max_drawdown_no_cost"],
                }
            )

            # With costs (standard)
            comparison_data.append(
                {
                    "Strategy": strategy_name,
                    "Scenario": "With Costs (0.1%)",
                    "Final Value": strategy_data["final_value_with_cost"],
                    "Annual Return (%)": strategy_data["annual_return_with_cost"],
                    "Sharpe Ratio": strategy_data["sharpe_with_cost"],
                    "Max Drawdown (%)": strategy_data["max_drawdown_with_cost"],
                }
            )

        df = pd.DataFrame(comparison_data)

        if output_path:
            df.to_csv(output_path, index=False)

        return df

    def plot_cost_impact(self, results_df: pd.DataFrame, save_path: str = None):
        """Plot the impact of transaction costs on performance."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Cost Impact by Rebalancing Frequency
        ax = axes[0, 0]
        pivot_data = results_df.pivot_table(
            values="cost_impact_pct",
            index="frequency_name",
            columns="cost_structure",
            aggfunc="mean",
        )
        pivot_data.plot(kind="bar", ax=ax)
        ax.set_title(
            "Cost Impact (%) by Rebalancing Frequency", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Rebalancing Frequency")
        ax.set_ylabel("Cost Impact (%)")
        ax.legend(title="Cost Structure")
        ax.grid(True, alpha=0.3)

        # 2. Sharpe Ratio Degradation
        ax = axes[0, 1]
        sharpe_data = results_df.groupby("frequency_name")[
            ["sharpe_no_cost", "sharpe_with_cost"]
        ].mean()
        sharpe_data.plot(kind="bar", ax=ax)
        ax.set_title(
            "Sharpe Ratio: With vs Without Costs", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Rebalancing Frequency")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend(["No Costs", "With Costs"])
        ax.grid(True, alpha=0.3)

        # 3. Total Transaction Costs
        ax = axes[1, 0]
        cost_data = results_df.pivot_table(
            values="total_transaction_costs",
            index="frequency_name",
            columns="cost_structure",
            aggfunc="mean",
        )
        cost_data.plot(kind="bar", ax=ax)
        ax.set_title(
            "Total Transaction Costs by Frequency", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Rebalancing Frequency")
        ax.set_ylabel("Total Costs ($)")
        ax.legend(title="Cost Structure")
        ax.grid(True, alpha=0.3)

        # 4. Optimal Rebalancing Frequency (Sharpe Ratio)
        ax = axes[1, 1]
        optimal_data = results_df.loc[
            results_df.groupby("cost_structure")["sharpe_with_cost"].idxmax()
        ]
        ax.barh(optimal_data["cost_structure"], optimal_data["sharpe_with_cost"])
        ax.set_title(
            "Optimal Sharpe Ratio by Cost Structure", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Cost Structure")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def generate_cost_report(
        self, results_df: pd.DataFrame, output_path: str = None
    ) -> str:
        """Generate a comprehensive transaction cost analysis report."""
        report = []
        report.append("=" * 80)
        report.append("TRANSACTION COST ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)
        report.append(
            f"Average Cost Impact: {results_df['cost_impact_pct'].mean():.2f}%"
        )
        report.append(f"Max Cost Impact: {results_df['cost_impact_pct'].max():.2f}%")
        report.append(f"Min Cost Impact: {results_df['cost_impact_pct'].min():.2f}%")
        report.append("")

        # Optimal rebalancing frequency
        report.append("OPTIMAL REBALANCING FREQUENCY")
        report.append("-" * 80)
        for cost_struct in results_df["cost_structure"].unique():
            subset = results_df[results_df["cost_structure"] == cost_struct]
            optimal_idx = subset["sharpe_with_cost"].idxmax()
            optimal_row = subset.loc[optimal_idx]
            report.append(f"\nCost Structure: {cost_struct}")
            report.append(f"  Optimal Frequency: {optimal_row['frequency_name']}")
            report.append(f"  Sharpe Ratio: {optimal_row['sharpe_with_cost']:.3f}")
            report.append(f"  Cost Impact: {optimal_row['cost_impact_pct']:.2f}%")
        report.append("")

        # Cost structure comparison
        report.append("COST STRUCTURE COMPARISON")
        report.append("-" * 80)
        cost_comparison = results_df.groupby("cost_structure").agg(
            {
                "cost_impact_pct": "mean",
                "sharpe_with_cost": "mean",
                "total_transaction_costs": "mean",
            }
        )
        report.append(cost_comparison.to_string())
        report.append("")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)

        return report_text

    def _calculate_sharpe(
        self, returns: np.ndarray, risk_free_rate: float = 0.045
    ) -> float:
        """Calculate Sharpe Ratio."""
        if len(returns) == 0:
            return 0.0
        annual_return = np.mean(returns) * 252
        annual_vol = np.std(returns) * np.sqrt(252)
        if annual_vol == 0:
            return 0.0
        return (annual_return - risk_free_rate) / annual_vol

    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown."""
        values_arr = np.array(values)
        peak = np.maximum.accumulate(values_arr)
        drawdown = (peak - values_arr) / peak
        return -np.max(drawdown) * 100  # Return as negative percentage


if __name__ == "__main__":
    print("Transaction Cost Analyzer module loaded successfully")
