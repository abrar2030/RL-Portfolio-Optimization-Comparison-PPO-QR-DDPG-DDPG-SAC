"""
Benchmark strategies implementation for comparison.

This module implements:
- Equal Weight (EW)
- Mean-Variance Optimization (MVO)
- Risk Parity (RP)
- Minimum Volatility Portfolio (MVP)
- Momentum Strategy (MOM)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class BenchmarkStrategies:
    """Collection of traditional portfolio optimization strategies."""
    
    def __init__(self, returns_data: pd.DataFrame, tickers: List[str]):
        """
        Initialize benchmark strategies.
        
        Args:
            returns_data: DataFrame with returns for each asset
            tickers: List of asset tickers
        """
        self.returns_data = returns_data
        self.tickers = tickers
        self.n_assets = len(tickers)
    
    def equal_weight(self) -> np.ndarray:
        """
        Equal Weight Portfolio.
        
        Returns:
            Array of equal weights
        """
        weights = np.ones(self.n_assets) / self.n_assets
        return weights
    
    def mean_variance_optimization(
        self, 
        target_return: float = None,
        risk_free_rate: float = 0.045
    ) -> np.ndarray:
        """
        Mean-Variance Optimization (Markowitz).
        Maximizes Sharpe Ratio.
        
        Args:
            target_return: Target portfolio return (if None, maximize Sharpe)
            risk_free_rate: Risk-free rate for Sharpe calculation
        
        Returns:
            Optimal portfolio weights
        """
        # Calculate mean returns and covariance
        mean_returns = self.returns_data.mean()
        cov_matrix = self.returns_data.cov()
        
        n_assets = len(mean_returns)
        
        # Objective: Minimize negative Sharpe ratio
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - risk_free_rate / 252) / portfolio_std
            return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum to 1
        ]
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        initial_weights = np.array([1 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else initial_weights
    
    def risk_parity(self) -> np.ndarray:
        """
        Risk Parity Portfolio.
        Allocates capital so each asset contributes equally to portfolio risk.
        
        Returns:
            Risk parity weights
        """
        cov_matrix = self.returns_data.cov()
        n_assets = len(cov_matrix)
        
        # Objective: Minimize difference in risk contributions
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Each asset should contribute 1/n of total risk
            target_risk = portfolio_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        initial_weights = np.array([1 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            risk_budget_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else initial_weights
    
    def minimum_volatility(self) -> np.ndarray:
        """
        Minimum Volatility Portfolio.
        Minimizes portfolio variance.
        
        Returns:
            Minimum volatility weights
        """
        cov_matrix = self.returns_data.cov()
        n_assets = len(cov_matrix)
        
        # Objective: Minimize portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        initial_weights = np.array([1 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else initial_weights
    
    def momentum(self, lookback_period: int = 20) -> np.ndarray:
        """
        Momentum Strategy.
        Allocates more weight to assets with higher recent returns.
        
        Args:
            lookback_period: Number of periods to calculate momentum
        
        Returns:
            Momentum-based weights
        """
        # Calculate momentum (average return over lookback period)
        recent_returns = self.returns_data.tail(lookback_period)
        momentum = recent_returns.mean()
        
        # Only invest in positive momentum assets
        momentum = momentum.clip(lower=0)
        
        # Normalize to sum to 1
        if momentum.sum() > 0:
            weights = momentum / momentum.sum()
        else:
            weights = np.ones(self.n_assets) / self.n_assets
        
        return weights.values


class BacktestBenchmark:
    """Backtest benchmark strategies."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float = 1000000,
        transaction_cost_pct: float = 0.001,
        rebalance_freq: int = 20  # Rebalance every 20 days
    ):
        """
        Initialize backtest.
        
        Args:
            df: Processed DataFrame with market data
            initial_amount: Initial portfolio value
            transaction_cost_pct: Transaction cost percentage
            rebalance_freq: Rebalancing frequency in days
        """
        self.df = df
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.rebalance_freq = rebalance_freq
        
        self.dates = df['Date'].unique()
        self.tickers = df['tic'].unique()
        self.n_assets = len(self.tickers)
    
    def backtest_strategy(
        self,
        strategy_name: str,
        lookback_window: int = 60
    ) -> Dict:
        """
        Backtest a specific strategy.
        
        Args:
            strategy_name: Name of strategy ('equal_weight', 'mvo', etc.)
            lookback_window: Lookback period for covariance estimation
        
        Returns:
            Dictionary with backtest results
        """
        portfolio_value = self.initial_amount
        portfolio_values = [portfolio_value]
        weights = np.ones(self.n_assets) / self.n_assets
        dates_list = [self.dates[0]]
        
        for i in range(1, len(self.dates)):
            current_date = self.dates[i]
            
            # Rebalance if needed
            if i % self.rebalance_freq == 0 and i >= lookback_window:
                # Get historical returns for lookback window
                lookback_dates = self.dates[i-lookback_window:i]
                returns_data = []
                
                for ticker in self.tickers:
                    ticker_data = self.df[
                        (self.df['tic'] == ticker) & 
                        (self.df['Date'].isin(lookback_dates))
                    ]['Close'].values
                    
                    if len(ticker_data) > 1:
                        ticker_returns = np.diff(ticker_data) / ticker_data[:-1]
                        returns_data.append(ticker_returns)
                    else:
                        returns_data.append(np.zeros(len(lookback_dates) - 1))
                
                returns_df = pd.DataFrame(
                    np.array(returns_data).T,
                    columns=self.tickers
                )
                
                # Calculate new weights
                benchmark = BenchmarkStrategies(returns_df, list(self.tickers))
                
                if strategy_name == 'equal_weight':
                    new_weights = benchmark.equal_weight()
                elif strategy_name == 'mvo':
                    new_weights = benchmark.mean_variance_optimization()
                elif strategy_name == 'risk_parity':
                    new_weights = benchmark.risk_parity()
                elif strategy_name == 'minimum_volatility':
                    new_weights = benchmark.minimum_volatility()
                elif strategy_name == 'momentum':
                    new_weights = benchmark.momentum()
                else:
                    new_weights = weights
                
                # Calculate transaction costs
                weight_changes = np.abs(new_weights - weights)
                transaction_cost = np.sum(weight_changes) * portfolio_value * self.transaction_cost_pct
                portfolio_value -= transaction_cost
                
                weights = new_weights
            
            # Calculate portfolio return
            prev_date = self.dates[i-1]
            portfolio_return = 0
            
            for j, ticker in enumerate(self.tickers):
                prev_price_data = self.df[
                    (self.df['tic'] == ticker) & (self.df['Date'] == prev_date)
                ]
                curr_price_data = self.df[
                    (self.df['tic'] == ticker) & (self.df['Date'] == current_date)
                ]
                
                if len(prev_price_data) > 0 and len(curr_price_data) > 0:
                    prev_price = prev_price_data.iloc[0]['Close']
                    curr_price = curr_price_data.iloc[0]['Close']
                    
                    if prev_price > 0:
                        asset_return = (curr_price - prev_price) / prev_price
                        portfolio_return += weights[j] * asset_return
            
            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)
            portfolio_values.append(portfolio_value)
            dates_list.append(current_date)
        
        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_value - self.initial_amount) / self.initial_amount
        days = len(returns)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        volatility = np.std(returns) * np.sqrt(252)
        risk_free_rate = 0.045
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum Drawdown
        portfolio_values_arr = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values_arr)
        drawdown = (peak - portfolio_values_arr) / peak
        max_drawdown = np.max(drawdown)
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # CVaR
        sorted_returns = np.sort(returns)
        cvar_index = int(0.05 * len(sorted_returns))
        cvar = np.mean(sorted_returns[:cvar_index]) if cvar_index > 0 else 0
        
        results = {
            'strategy': strategy_name,
            'portfolio_values': portfolio_values,
            'dates': dates_list,
            'final_value': portfolio_value,
            'total_return': total_return,
            'annual_return': annual_return * 100,  # Percentage
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': -max_drawdown * 100,  # Percentage
            'cvar_5': cvar * 100,  # Percentage
            'volatility': volatility * 100  # Percentage
        }
        
        return results


if __name__ == "__main__":
    print("Benchmark strategies module loaded successfully")
