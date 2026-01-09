"""
Initialize the src package.
"""

from .data_processor import DataProcessor
from .environment import PortfolioEnv
from .agents import DDPGAgent, QRDDPGAgent
from .benchmarks import BenchmarkStrategies, BacktestBenchmark
from .utils import calculate_portfolio_metrics, normalize_weights

__version__ = "1.0.0"
__author__ = "Abrar Ahmed"

__all__ = [
    "DataProcessor",
    "PortfolioEnv",
    "DDPGAgent",
    "QRDDPGAgent",
    "BenchmarkStrategies",
    "BacktestBenchmark",
    "calculate_portfolio_metrics",
    "normalize_weights",
]
