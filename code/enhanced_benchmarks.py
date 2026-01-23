"""
Enhanced Benchmark Strategies Implementation.

Extended to include:
- 60/40 Portfolio
- All-Weather Portfolio
- Minimum Correlation Portfolio
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")


class EnhancedBenchmarkStrategies:
    """Extended collection of portfolio optimization strategies."""

    def __init__(
        self, returns_data: pd.DataFrame, tickers: List[str], asset_classes: Dict = None
    ):
        """
        Initialize enhanced benchmark strategies.

        Args:
            returns_data: DataFrame with returns for each asset
            tickers: List of asset tickers
            asset_classes: Dictionary mapping tickers to asset classes
        """
        self.returns_data = returns_data
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.asset_classes = asset_classes or {}

    def equal_weight(self) -> np.ndarray:
        """Equal Weight Portfolio."""
        weights = np.ones(self.n_assets) / self.n_assets
        return weights

    def mean_variance_optimization(
        self, target_return: float = None, risk_free_rate: float = 0.045
    ) -> np.ndarray:
        """Mean-Variance Optimization (Markowitz)."""
        mean_returns = self.returns_data.mean()
        cov_matrix = self.returns_data.cov()
        n_assets = len(mean_returns)

        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - risk_free_rate / 252) / portfolio_std
            return -sharpe

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            negative_sharpe,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else initial_weights

    def risk_parity(self) -> np.ndarray:
        """Risk Parity Portfolio."""
        cov_matrix = self.returns_data.cov()
        n_assets = len(cov_matrix)

        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            target_risk = portfolio_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            risk_budget_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else initial_weights

    def minimum_volatility(self) -> np.ndarray:
        """Minimum Volatility Portfolio."""
        cov_matrix = self.returns_data.cov()
        n_assets = len(cov_matrix)

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            portfolio_variance,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else initial_weights

    def momentum(self, lookback_period: int = 20) -> np.ndarray:
        """Momentum Strategy."""
        recent_returns = self.returns_data.tail(lookback_period)
        momentum = recent_returns.mean()
        momentum = momentum.clip(lower=0)

        if momentum.sum() > 0:
            weights = momentum / momentum.sum()
        else:
            weights = np.ones(self.n_assets) / self.n_assets

        return weights.values

    def sixty_forty(self) -> np.ndarray:
        """
        60/40 Portfolio: 60% Stocks, 40% Bonds.
        Classic balanced portfolio.
        """
        weights = np.zeros(self.n_assets)

        # Identify stocks and bonds
        stock_indices = []
        bond_indices = []

        for i, ticker in enumerate(self.tickers):
            asset_class = self.asset_classes.get(ticker, "")
            if asset_class == "equities":
                stock_indices.append(i)
            elif asset_class == "fixed_income":
                bond_indices.append(i)

        # Allocate 60% to stocks, 40% to bonds
        if len(stock_indices) > 0:
            stock_weight = 0.60 / len(stock_indices)
            for i in stock_indices:
                weights[i] = stock_weight

        if len(bond_indices) > 0:
            bond_weight = 0.40 / len(bond_indices)
            for i in bond_indices:
                weights[i] = bond_weight

        # Normalize if needed
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(self.n_assets) / self.n_assets

        return weights

    def all_weather(self) -> np.ndarray:
        """
        All-Weather Portfolio (Ray Dalio).
        Designed to perform well in all economic environments.

        Classic allocation:
        - 30% Stocks
        - 40% Long-term bonds
        - 15% Intermediate bonds
        - 7.5% Gold
        - 7.5% Commodities
        """
        weights = np.zeros(self.n_assets)

        for i, ticker in enumerate(self.tickers):
            asset_class = self.asset_classes.get(ticker, "")

            if asset_class == "equities":
                # 30% stocks
                weights[i] = 0.30 / max(
                    1,
                    sum(
                        1
                        for t in self.tickers
                        if self.asset_classes.get(t) == "equities"
                    ),
                )
            elif asset_class == "fixed_income":
                # 55% bonds (40% long + 15% intermediate)
                if "TLT" in ticker:  # Long-term
                    weights[i] = 0.40
                elif "IEF" in ticker:  # Intermediate
                    weights[i] = 0.15
                else:
                    # Distribute among other bonds
                    bond_count = sum(
                        1
                        for t in self.tickers
                        if self.asset_classes.get(t) == "fixed_income"
                    )
                    weights[i] = 0.55 / max(1, bond_count)
            elif asset_class == "commodities":
                # 15% commodities
                commodity_count = sum(
                    1
                    for t in self.tickers
                    if self.asset_classes.get(t) == "commodities"
                )
                if "GC=F" in ticker:  # Gold gets slightly more
                    weights[i] = 0.075
                else:
                    weights[i] = 0.075 / max(1, commodity_count - 1)

        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(self.n_assets) / self.n_assets

        return weights

    def minimum_correlation(self) -> np.ndarray:
        """
        Minimum Correlation Portfolio.
        Minimizes average correlation between assets.
        """
        corr_matrix = self.returns_data.corr()
        n_assets = len(corr_matrix)

        def average_correlation(weights):
            # Calculate weighted average correlation
            weighted_corr = 0
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    weighted_corr += weights[i] * weights[j] * corr_matrix.iloc[i, j]
            return weighted_corr

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            average_correlation,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else initial_weights


if __name__ == "__main__":
    print("Enhanced Benchmark Strategies module loaded successfully")
