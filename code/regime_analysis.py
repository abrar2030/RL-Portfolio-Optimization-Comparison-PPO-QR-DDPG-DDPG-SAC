"""
Market Regime Analysis Module.

Classifies market conditions and analyzes algorithm performance
across different regimes (bull, bear, sideways).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import yaml
from pathlib import Path


class MarketRegimeAnalyzer:
    """Analyze portfolio performance across market regimes."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize regime analyzer."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.regime_config = self.config["regime_analysis"]
        self.output_dir = Path(self.regime_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def identify_regimes_vix(
        self, market_data: pd.DataFrame, vix_column: str = "VIX"
    ) -> pd.DataFrame:
        """
        Identify market regimes using VIX thresholds.

        Args:
            market_data: DataFrame with market data including VIX
            vix_column: Column name for VIX data

        Returns:
            DataFrame with regime labels
        """
        df = market_data.copy()

        bull_threshold = self.regime_config["regime_definitions"]["bull"][
            "vix_threshold"
        ]
        bear_threshold = self.regime_config["regime_definitions"]["bear"][
            "vix_threshold"
        ]

        def classify_regime(vix):
            if vix < bull_threshold:
                return "bull"
            elif vix > bear_threshold:
                return "bear"
            else:
                return "sideways"

        df["regime"] = df[vix_column].apply(classify_regime)
        return df

    def identify_regimes_trend(
        self,
        market_data: pd.DataFrame,
        price_column: str = "Close",
        short_window: int = 50,
        long_window: int = 200,
    ) -> pd.DataFrame:
        """
        Identify market regimes using moving average crossover.

        Args:
            market_data: DataFrame with price data
            price_column: Column name for price data
            short_window: Short MA window
            long_window: Long MA window

        Returns:
            DataFrame with regime labels
        """
        df = market_data.copy()

        df["SMA_short"] = df[price_column].rolling(window=short_window).mean()
        df["SMA_long"] = df[price_column].rolling(window=long_window).mean()

        def classify_trend(row):
            if pd.isna(row["SMA_short"]) or pd.isna(row["SMA_long"]):
                return "sideways"

            diff = (row["SMA_short"] - row["SMA_long"]) / row["SMA_long"]

            if diff > 0.05:
                return "bull"
            elif diff < -0.05:
                return "bear"
            else:
                return "sideways"

        df["regime"] = df.apply(classify_trend, axis=1)
        return df

    def identify_regimes_returns(
        self,
        market_data: pd.DataFrame,
        returns_column: str = "returns",
        lookback_window: int = 60,
    ) -> pd.DataFrame:
        """
        Identify market regimes using return-based clustering.

        Args:
            market_data: DataFrame with returns data
            returns_column: Column name for returns
            lookback_window: Lookback period for regime classification

        Returns:
            DataFrame with regime labels
        """
        df = market_data.copy()

        bull_return = self.regime_config["regime_definitions"]["bull"][
            "return_threshold"
        ]
        bear_return = self.regime_config["regime_definitions"]["bear"][
            "return_threshold"
        ]

        # Calculate rolling return
        df["rolling_return"] = (
            df[returns_column].rolling(window=lookback_window).mean() * 252
        )

        def classify_return_regime(ret):
            if pd.isna(ret):
                return "sideways"
            if ret > bull_return:
                return "bull"
            elif ret < bear_return:
                return "bear"
            else:
                return "sideways"

        df["regime"] = df["rolling_return"].apply(classify_return_regime)
        return df

    def analyze_performance_by_regime(
        self, strategy_results: Dict[str, pd.DataFrame], regime_labels: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze strategy performance broken down by market regime.

        Args:
            strategy_results: Dictionary mapping strategy names to result DataFrames
            regime_labels: DataFrame with regime classifications

        Returns:
            DataFrame with performance metrics by strategy and regime
        """
        analysis_results = []

        for strategy_name, results_df in strategy_results.items():
            # Merge with regime labels
            merged = pd.merge(
                results_df, regime_labels[["date", "regime"]], on="date", how="inner"
            )

            # Calculate metrics for each regime
            for regime in ["bull", "bear", "sideways"]:
                regime_data = merged[merged["regime"] == regime]

                if len(regime_data) > 0:
                    metrics = self._calculate_regime_metrics(regime_data)
                    metrics["strategy"] = strategy_name
                    metrics["regime"] = regime
                    metrics["n_periods"] = len(regime_data)
                    analysis_results.append(metrics)

        return pd.DataFrame(analysis_results)

    def compare_algorithms_by_regime(
        self, performance_df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Identify which algorithm performs best in each regime.

        Args:
            performance_df: DataFrame with performance by strategy and regime

        Returns:
            Dictionary mapping regimes to best-performing algorithms
        """
        best_performers = {}

        for regime in ["bull", "bear", "sideways"]:
            regime_data = performance_df[performance_df["regime"] == regime]

            # Find best Sharpe ratio
            best_idx = regime_data["sharpe_ratio"].idxmax()
            best_strategy = regime_data.loc[best_idx, "strategy"]
            best_sharpe = regime_data.loc[best_idx, "sharpe_ratio"]

            best_performers[regime] = {
                "strategy": best_strategy,
                "sharpe_ratio": best_sharpe,
                "annual_return": regime_data.loc[best_idx, "annual_return"],
                "max_drawdown": regime_data.loc[best_idx, "max_drawdown"],
            }

        return best_performers

    def plot_regime_performance(
        self, performance_df: pd.DataFrame, save_path: str = None
    ):
        """
        Plot performance comparison across regimes.

        Args:
            performance_df: DataFrame with performance metrics
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Sharpe Ratio by Regime
        ax = axes[0, 0]
        pivot_sharpe = performance_df.pivot_table(
            values="sharpe_ratio", index="strategy", columns="regime", aggfunc="mean"
        )
        pivot_sharpe.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Sharpe Ratio by Market Regime", fontsize=14, fontweight="bold")
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend(title="Regime")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 2. Annual Return by Regime
        ax = axes[0, 1]
        pivot_return = performance_df.pivot_table(
            values="annual_return", index="strategy", columns="regime", aggfunc="mean"
        )
        pivot_return.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Annual Return by Market Regime", fontsize=14, fontweight="bold")
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Annual Return (%)")
        ax.legend(title="Regime")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 3. Max Drawdown by Regime
        ax = axes[1, 0]
        pivot_dd = performance_df.pivot_table(
            values="max_drawdown", index="strategy", columns="regime", aggfunc="mean"
        )
        pivot_dd.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Max Drawdown by Market Regime", fontsize=14, fontweight="bold")
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Max Drawdown (%)")
        ax.legend(title="Regime")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 4. Heatmap of Sharpe Ratios
        ax = axes[1, 1]
        sns.heatmap(
            pivot_sharpe,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=1.0,
            ax=ax,
            cbar_kws={"label": "Sharpe Ratio"},
        )
        ax.set_title("Sharpe Ratio Heatmap by Regime", fontsize=14, fontweight="bold")
        ax.set_xlabel("Market Regime")
        ax.set_ylabel("Strategy")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_regime_timeline(
        self,
        regime_df: pd.DataFrame,
        price_data: pd.DataFrame = None,
        save_path: str = None,
    ):
        """
        Plot timeline showing regime transitions.

        Args:
            regime_df: DataFrame with regime labels
            price_data: Optional price data to overlay
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        # Color mapping
        regime_colors = {"bull": "green", "bear": "red", "sideways": "gray"}

        # 1. Regime timeline
        ax = axes[0]
        for regime in ["bull", "bear", "sideways"]:
            regime_periods = regime_df[regime_df["regime"] == regime]
            ax.scatter(
                regime_periods["date"],
                [regime] * len(regime_periods),
                c=regime_colors[regime],
                s=10,
                alpha=0.6,
                label=regime.capitalize(),
            )

        ax.set_ylabel("Market Regime", fontsize=12)
        ax.set_title("Market Regime Timeline", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Price overlay (if provided)
        if price_data is not None:
            ax = axes[1]
            ax.plot(
                price_data["date"], price_data["Close"], linewidth=1.5, color="black"
            )

            # Shade regime periods
            for regime in ["bull", "bear", "sideways"]:
                regime_periods = regime_df[regime_df["regime"] == regime]
                for date in regime_periods["date"]:
                    ax.axvspan(date, date, alpha=0.1, color=regime_colors[regime])

            ax.set_ylabel("Price", fontsize=12)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_title(
                "Price Chart with Regime Overlay", fontsize=14, fontweight="bold"
            )
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def generate_regime_report(
        self,
        performance_df: pd.DataFrame,
        best_performers: Dict,
        output_path: str = None,
    ) -> str:
        """Generate comprehensive regime analysis report."""
        report = []
        report.append("=" * 80)
        report.append("MARKET REGIME ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Regime distribution
        report.append("REGIME DISTRIBUTION")
        report.append("-" * 80)
        regime_counts = performance_df.groupby("regime")["n_periods"].first()
        total_periods = regime_counts.sum()

        for regime, count in regime_counts.items():
            pct = count / total_periods * 100
            report.append(f"{regime.capitalize()}: {count} periods ({pct:.1f}%)")
        report.append("")

        # Best performers by regime
        report.append("BEST PERFORMING STRATEGIES BY REGIME")
        report.append("-" * 80)
        for regime, perf in best_performers.items():
            report.append(f"\n{regime.upper()} MARKET:")
            report.append(f"  Best Strategy: {perf['strategy']}")
            report.append(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
            report.append(f"  Annual Return: {perf['annual_return']:.2f}%")
            report.append(f"  Max Drawdown: {perf['max_drawdown']:.2f}%")
        report.append("")

        # Performance comparison table
        report.append("DETAILED PERFORMANCE BY REGIME")
        report.append("-" * 80)
        summary = performance_df.pivot_table(
            values=["sharpe_ratio", "annual_return", "max_drawdown"],
            index="strategy",
            columns="regime",
            aggfunc="mean",
        )
        report.append(summary.to_string())
        report.append("")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)

        return report_text

    def _calculate_regime_metrics(self, regime_data: pd.DataFrame) -> Dict:
        """Calculate performance metrics for a specific regime."""
        returns = regime_data["returns"].values if "returns" in regime_data else []

        if len(returns) == 0:
            return {
                "annual_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "volatility": 0,
            }

        annual_return = np.mean(returns) * 252 * 100
        volatility = np.std(returns) * np.sqrt(252) * 100
        sharpe = (
            (np.mean(returns) * 252 - 0.045) / (np.std(returns) * np.sqrt(252))
            if np.std(returns) > 0
            else 0
        )

        # Max drawdown
        if "portfolio_value" in regime_data:
            values = regime_data["portfolio_value"].values
            peak = np.maximum.accumulate(values)
            drawdown = (peak - values) / peak
            max_dd = -np.max(drawdown) * 100
        else:
            max_dd = 0

        return {
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "volatility": volatility,
        }


if __name__ == "__main__":
    print("Market Regime Analyzer module loaded successfully")
