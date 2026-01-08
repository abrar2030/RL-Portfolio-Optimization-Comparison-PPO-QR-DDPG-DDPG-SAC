"""
Custom Portfolio Environment for RL Training.

This module implements the Markov Decision Process (MDP) formulation
with risk-aware reward function.
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class PortfolioEnv(gym.Env):
    """
    Custom Environment for Portfolio Optimization.
    
    State Space: Asset prices, technical indicators, macro factors, 
                 current portfolio weights, portfolio value
    Action Space: Continuous portfolio weight changes
    Reward: Log return - max drawdown penalty - transaction costs
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float = 1000000,
        transaction_cost_pct: float = 0.001,
        max_drawdown_penalty: float = 0.5,
        hmax: int = 100,
        print_verbosity: int = 5,
        turbulence_threshold: float = None
    ):
        """
        Initialize the Portfolio Environment.
        
        Args:
            df: Processed DataFrame with market data
            initial_amount: Initial portfolio value
            transaction_cost_pct: Transaction cost percentage
            max_drawdown_penalty: Lambda coefficient for drawdown penalty
            hmax: Maximum number of shares per trade
            print_verbosity: How often to print progress
            turbulence_threshold: Threshold for turbulence-based risk management
        """
        self.df = df.copy()
        self.df = self.df.sort_values(['Date', 'tic']).reset_index(drop=True)
        
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.max_drawdown_penalty = max_drawdown_penalty
        self.hmax = hmax
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        
        # Get unique dates and tickers
        self.dates = self.df['Date'].unique()
        self.tickers = self.df['tic'].unique()
        self.n_stocks = len(self.tickers)
        
        # Calculate state dimension
        self.state_dim = 1 + self.n_stocks + self.n_stocks * 6  # portfolio value + weights + features per stock
        
        # Action space: portfolio weight changes for each asset
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.n_stocks,),
            dtype=np.float32
        )
        
        # State space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Initialize episode variables
        self.current_step = 0
        self.portfolio_value = initial_amount
        self.initial_portfolio_value = initial_amount
        self.portfolio_weights = np.zeros(self.n_stocks)
        self.cash_balance = initial_amount
        
        # For tracking performance
        self.portfolio_values = [initial_amount]
        self.portfolio_returns = [0]
        self.max_portfolio_value = initial_amount
        self.actions_memory = []
        self.date_memory = [self.dates[0]]
        
        print(f"Environment initialized with {self.n_stocks} assets")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.n_stocks}")
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        self.initial_portfolio_value = self.initial_amount
        self.portfolio_weights = np.zeros(self.n_stocks)
        self.cash_balance = self.initial_amount
        
        self.portfolio_values = [self.initial_amount]
        self.portfolio_returns = [0]
        self.max_portfolio_value = self.initial_amount
        self.actions_memory = []
        self.date_memory = [self.dates[0]]
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Construct the state vector.
        
        Returns:
            State vector containing portfolio value, weights, and market features
        """
        if self.current_step >= len(self.dates):
            return np.zeros(self.state_dim)
        
        current_date = self.dates[self.current_step]
        current_data = self.df[self.df['Date'] == current_date]
        
        # Portfolio value (normalized)
        normalized_value = self.portfolio_value / self.initial_amount
        
        # Portfolio weights
        weights = self.portfolio_weights.copy()
        
        # Market features for each asset
        features = []
        for ticker in self.tickers:
            ticker_data = current_data[current_data['tic'] == ticker]
            
            if len(ticker_data) == 0:
                # If no data available, use zeros
                features.extend([0, 0, 0, 0, 0, 0])
            else:
                row = ticker_data.iloc[0]
                # Normalize features
                close_price = row['Close'] / 100 if 'Close' in row else 0
                macd = row['macd'] / 10 if 'macd' in row else 0
                rsi = row['rsi'] / 100 if 'rsi' in row else 0
                cci = row['cci'] / 100 if 'cci' in row else 0
                dx = row['dx'] / 100 if 'dx' in row else 0
                boll_ub = row['boll_ub'] / 100 if 'boll_ub' in row else 0
                
                features.extend([close_price, macd, rsi, cci, dx, boll_ub])
        
        # Combine all state components
        state = np.array([normalized_value] + list(weights) + features, dtype=np.float32)
        
        return state
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step in the environment.
        
        Args:
            actions: Portfolio weight changes
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Ensure actions are valid
        actions = np.clip(actions, -1, 1)
        
        # Get current data
        current_date = self.dates[self.current_step]
        current_data = self.df[self.df['Date'] == current_date]
        
        # Calculate new weights
        new_weights = self.portfolio_weights + actions
        new_weights = np.clip(new_weights, 0, 1)
        
        # Normalize weights to sum to 1
        weight_sum = np.sum(new_weights)
        if weight_sum > 0:
            new_weights = new_weights / weight_sum
        else:
            new_weights = np.ones(self.n_stocks) / self.n_stocks
        
        # Calculate transaction costs
        weight_changes = np.abs(new_weights - self.portfolio_weights)
        transaction_cost = np.sum(weight_changes) * self.portfolio_value * self.transaction_cost_pct
        
        # Update weights
        old_weights = self.portfolio_weights.copy()
        self.portfolio_weights = new_weights
        
        # Move to next time step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.dates) - 1
        
        if not done:
            # Get next period data
            next_date = self.dates[self.current_step]
            next_data = self.df[self.df['Date'] == next_date]
            
            # Calculate portfolio return
            portfolio_return = 0
            for i, ticker in enumerate(self.tickers):
                current_price_data = current_data[current_data['tic'] == ticker]
                next_price_data = next_data[next_data['tic'] == ticker]
                
                if len(current_price_data) > 0 and len(next_price_data) > 0:
                    current_price = current_price_data.iloc[0]['Close']
                    next_price = next_price_data.iloc[0]['Close']
                    
                    if current_price > 0:
                        asset_return = (next_price - current_price) / current_price
                        portfolio_return += self.portfolio_weights[i] * asset_return
            
            # Update portfolio value
            old_portfolio_value = self.portfolio_value
            self.portfolio_value = self.portfolio_value * (1 + portfolio_return) - transaction_cost
            
            # Update max portfolio value
            if self.portfolio_value > self.max_portfolio_value:
                self.max_portfolio_value = self.portfolio_value
            
            # Calculate log return
            if old_portfolio_value > 0:
                log_return = np.log(self.portfolio_value / old_portfolio_value)
            else:
                log_return = 0
            
            # Calculate maximum drawdown
            if self.max_portfolio_value > 0:
                current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            else:
                current_drawdown = 0
            
            # Calculate reward with drawdown penalty
            reward = log_return - self.max_drawdown_penalty * current_drawdown - (transaction_cost / self.initial_amount)
            
            # Store metrics
            self.portfolio_values.append(self.portfolio_value)
            self.portfolio_returns.append(portfolio_return)
            self.actions_memory.append(actions)
            self.date_memory.append(next_date)
        else:
            reward = 0
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'date': self.dates[self.current_step] if self.current_step < len(self.dates) else None,
            'transaction_cost': transaction_cost
        }
        
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment state."""
        if self.current_step % self.print_verbosity == 0:
            print(f"Step: {self.current_step}, Portfolio Value: ${self.portfolio_value:,.2f}")
    
    def save_portfolio_values(self) -> pd.DataFrame:
        """
        Save portfolio values to DataFrame.
        
        Returns:
            DataFrame with dates and portfolio values
        """
        df = pd.DataFrame({
            'date': self.date_memory,
            'portfolio_value': self.portfolio_values
        })
        return df
    
    def get_portfolio_metrics(self) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        returns = np.array(self.portfolio_returns[1:])  # Exclude first zero return
        
        if len(returns) == 0:
            return {}
        
        # Calculate metrics
        total_return = (self.portfolio_value - self.initial_amount) / self.initial_amount
        
        # Annualized return (assuming 252 trading days)
        days = len(returns)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe Ratio (assume risk-free rate = 4.5%)
        risk_free_rate = 0.045
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum Drawdown
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # CVaR (5%)
        sorted_returns = np.sort(returns)
        cvar_index = int(0.05 * len(sorted_returns))
        cvar = np.mean(sorted_returns[:cvar_index]) if cvar_index > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': -max_drawdown,
            'cvar_5': cvar
        }
        
        return metrics


if __name__ == "__main__":
    print("Portfolio Environment module loaded successfully")
