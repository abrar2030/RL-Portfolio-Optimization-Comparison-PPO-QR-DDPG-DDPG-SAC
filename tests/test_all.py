"""
Unit tests for data processor module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2020-12-31',
                'train_start': '2020-01-01',
                'train_end': '2020-06-30',
                'test_start': '2020-07-01',
                'test_end': '2020-12-31',
                'assets': {
                    'equities': ['AAPL', 'MSFT'],
                    'cryptocurrencies': [],
                    'commodities': [],
                    'fixed_income': []
                },
                'technical_indicators': ['macd', 'rsi'],
                'macro_factors': []
            }
        }
        
        self.processor = DataProcessor(self.config)
    
    def test_initialization(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.config, self.config)
    
    def test_fetch_data(self):
        """Test data fetching."""
        # This test requires internet connection
        try:
            data = self.processor.fetch_data()
            self.assertIsNotNone(data)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
        except Exception as e:
            self.skipTest(f"Data fetching failed (might be network issue): {e}")
    
    def test_technical_indicators(self):
        """Test technical indicator calculation."""
        # Create dummy data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        dummy_data = []
        
        for ticker in ['AAPL', 'MSFT']:
            for date in dates:
                dummy_data.append({
                    'Date': date,
                    'tic': ticker,
                    'Open': 100 + np.random.randn(),
                    'High': 105 + np.random.randn(),
                    'Low': 95 + np.random.randn(),
                    'Close': 100 + np.random.randn(),
                    'Volume': 1000000
                })
        
        self.processor.data = pd.DataFrame(dummy_data)
        
        # Calculate indicators
        processed = self.processor.calculate_technical_indicators()
        
        self.assertIn('macd', processed.columns)
        self.assertIn('rsi', processed.columns)
        self.assertGreater(len(processed), 0)


class TestEnvironment(unittest.TestCase):
    """Test cases for Portfolio Environment."""
    
    def setUp(self):
        """Set up test environment."""
        from environment import PortfolioEnv
        
        # Create dummy data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        dummy_data = []
        
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        for ticker in tickers:
            for date in dates:
                dummy_data.append({
                    'Date': date,
                    'tic': ticker,
                    'Close': 100 + np.random.randn() * 10,
                    'macd': np.random.randn(),
                    'rsi': 50 + np.random.randn() * 20,
                    'cci': np.random.randn() * 50,
                    'dx': 20 + np.random.randn() * 10,
                    'boll_ub': 110,
                    'boll_lb': 90
                })
        
        self.df = pd.DataFrame(dummy_data)
        self.env = PortfolioEnv(self.df, initial_amount=100000)
    
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        
        self.assertIsNotNone(state)
        self.assertEqual(state.shape, (self.env.state_dim,))
        self.assertEqual(self.env.portfolio_value, self.env.initial_amount)
    
    def test_step(self):
        """Test environment step."""
        state = self.env.reset()
        action = np.random.randn(self.env.n_stocks)
        
        next_state, reward, done, info = self.env.step(action)
        
        self.assertEqual(next_state.shape, state.shape)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
    
    def test_action_space(self):
        """Test action space."""
        self.assertEqual(self.env.action_space.shape[0], self.env.n_stocks)
    
    def test_observation_space(self):
        """Test observation space."""
        self.assertEqual(self.env.observation_space.shape[0], self.env.state_dim)


class TestAgents(unittest.TestCase):
    """Test cases for DRL agents."""
    
    def test_ddpg_initialization(self):
        """Test DDPG agent initialization."""
        from agents import DDPGAgent
        
        agent = DDPGAgent(state_dim=10, action_dim=5)
        
        self.assertIsNotNone(agent.actor)
        self.assertIsNotNone(agent.critic)
        self.assertEqual(agent.state_dim, 10)
        self.assertEqual(agent.action_dim, 5)
    
    def test_qr_ddpg_initialization(self):
        """Test QR-DDPG agent initialization."""
        from agents import QRDDPGAgent
        
        agent = QRDDPGAgent(state_dim=10, action_dim=5, n_quantiles=50)
        
        self.assertIsNotNone(agent.actor)
        self.assertIsNotNone(agent.critic)
        self.assertEqual(agent.n_quantiles, 50)
    
    def test_select_action(self):
        """Test action selection."""
        from agents import DDPGAgent
        
        agent = DDPGAgent(state_dim=10, action_dim=5)
        state = np.random.randn(10)
        
        action = agent.select_action(state, noise=0.1)
        
        self.assertEqual(action.shape, (5,))
        self.assertTrue(np.all(action >= -1) and np.all(action <= 1))


class TestBenchmarks(unittest.TestCase):
    """Test cases for benchmark strategies."""
    
    def setUp(self):
        """Set up test data."""
        # Create dummy returns data
        n_days = 100
        n_assets = 5
        
        returns = np.random.randn(n_days, n_assets) * 0.01
        self.returns_df = pd.DataFrame(
            returns,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        self.tickers = list(self.returns_df.columns)
    
    def test_equal_weight(self):
        """Test equal weight strategy."""
        from benchmarks import BenchmarkStrategies
        
        benchmark = BenchmarkStrategies(self.returns_df, self.tickers)
        weights = benchmark.equal_weight()
        
        self.assertEqual(len(weights), len(self.tickers))
        self.assertAlmostEqual(np.sum(weights), 1.0)
        self.assertTrue(np.all(weights >= 0))
    
    def test_mvo(self):
        """Test mean-variance optimization."""
        from benchmarks import BenchmarkStrategies
        
        benchmark = BenchmarkStrategies(self.returns_df, self.tickers)
        weights = benchmark.mean_variance_optimization()
        
        self.assertEqual(len(weights), len(self.tickers))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        self.assertTrue(np.all(weights >= -0.01))  # Allow small numerical errors
    
    def test_minimum_volatility(self):
        """Test minimum volatility strategy."""
        from benchmarks import BenchmarkStrategies
        
        benchmark = BenchmarkStrategies(self.returns_df, self.tickers)
        weights = benchmark.minimum_volatility()
        
        self.assertEqual(len(weights), len(self.tickers))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_calculate_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        from utils import calculate_portfolio_metrics
        
        portfolio_values = [1000000, 1010000, 1020000, 1015000, 1025000]
        metrics = calculate_portfolio_metrics(portfolio_values)
        
        self.assertIn('annual_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
    
    def test_normalize_weights(self):
        """Test weight normalization."""
        from utils import normalize_weights
        
        weights = np.array([0.3, 0.5, 0.2])
        normalized = normalize_weights(weights)
        
        self.assertAlmostEqual(np.sum(normalized), 1.0)
        self.assertTrue(np.all(normalized >= 0))


if __name__ == '__main__':
    unittest.main()
