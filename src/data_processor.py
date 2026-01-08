"""
Data preprocessing and feature engineering module.

This module handles:
1. Data fetching from Yahoo Finance
2. Technical indicator calculation
3. Feature engineering for the RL environment
4. Data cleaning and normalization
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Process financial data for RL training."""
    
    def __init__(self, config: Dict):
        """
        Initialize DataProcessor.
        
        Args:
            config: Configuration dictionary with data parameters
        """
        self.config = config
        self.data = None
        self.processed_data = None
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical data for all assets.
        
        Returns:
            DataFrame with OHLCV data for all assets
        """
        print("Fetching data from Yahoo Finance...")
        
        # Combine all asset tickers
        all_assets = []
        for asset_class in ['equities', 'cryptocurrencies', 'commodities', 'fixed_income']:
            all_assets.extend(self.config['data']['assets'][asset_class])
        
        # Add macro factors
        all_assets.extend(self.config['data']['macro_factors'])
        
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        
        # Download data
        data_dict = {}
        for ticker in all_assets:
            try:
                print(f"Downloading {ticker}...")
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    df['tic'] = ticker
                    data_dict[ticker] = df
                else:
                    print(f"Warning: No data for {ticker}")
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
        
        # Combine all data
        data_list = []
        for ticker, df in data_dict.items():
            df = df.reset_index()
            df['tic'] = ticker
            data_list.append(df)
        
        self.data = pd.concat(data_list, ignore_index=True)
        self.data = self.data.sort_values(['Date', 'tic']).reset_index(drop=True)
        
        print(f"Data fetched: {len(self.data)} rows, {self.data['tic'].nunique()} tickers")
        return self.data
    
    def calculate_technical_indicators(self) -> pd.DataFrame:
        """
        Calculate technical indicators for each asset.
        
        Returns:
            DataFrame with technical indicators
        """
        print("Calculating technical indicators...")
        
        df = self.data.copy()
        unique_tickers = df['tic'].unique()
        
        processed_list = []
        for ticker in unique_tickers:
            ticker_df = df[df['tic'] == ticker].copy()
            
            # MACD
            exp1 = ticker_df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = ticker_df['Close'].ewm(span=26, adjust=False).mean()
            ticker_df['macd'] = exp1 - exp2
            ticker_df['macd_signal'] = ticker_df['macd'].ewm(span=9, adjust=False).mean()
            ticker_df['macd_diff'] = ticker_df['macd'] - ticker_df['macd_signal']
            
            # RSI
            delta = ticker_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            ticker_df['rsi'] = 100 - (100 / (1 + rs))
            
            # CCI (Commodity Channel Index)
            tp = (ticker_df['High'] + ticker_df['Low'] + ticker_df['Close']) / 3
            sma = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            ticker_df['cci'] = (tp - sma) / (0.015 * mad)
            
            # DX (Directional Index)
            high_diff = ticker_df['High'].diff()
            low_diff = -ticker_df['Low'].diff()
            
            pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            tr1 = ticker_df['High'] - ticker_df['Low']
            tr2 = abs(ticker_df['High'] - ticker_df['Close'].shift())
            tr3 = abs(ticker_df['Low'] - ticker_df['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(window=14).mean()
            pos_di = 100 * (pos_dm.rolling(window=14).mean() / atr)
            neg_di = 100 * (neg_dm.rolling(window=14).mean() / atr)
            
            ticker_df['dx'] = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
            
            # Bollinger Bands
            sma_20 = ticker_df['Close'].rolling(window=20).mean()
            std_20 = ticker_df['Close'].rolling(window=20).std()
            ticker_df['boll_ub'] = sma_20 + 2 * std_20
            ticker_df['boll_lb'] = sma_20 - 2 * std_20
            
            processed_list.append(ticker_df)
        
        self.processed_data = pd.concat(processed_list, ignore_index=True)
        self.processed_data = self.processed_data.sort_values(['Date', 'tic']).reset_index(drop=True)
        
        # Forward fill and backward fill NaN values
        self.processed_data = self.processed_data.fillna(method='ffill').fillna(method='bfill')
        
        print("Technical indicators calculated successfully")
        return self.processed_data
    
    def add_turbulence_index(self) -> pd.DataFrame:
        """
        Add market turbulence index to the data.
        
        Returns:
            DataFrame with turbulence index
        """
        print("Calculating turbulence index...")
        
        df = self.processed_data.copy()
        
        # Calculate returns
        df['returns'] = df.groupby('tic')['Close'].pct_change()
        
        # Pivot returns to have tickers as columns
        returns_pivot = df.pivot(index='Date', columns='tic', values='returns')
        
        # Calculate covariance matrix
        cov_matrix = returns_pivot.cov()
        
        # Calculate turbulence for each date
        turbulence_list = []
        for date in returns_pivot.index:
            current_returns = returns_pivot.loc[date].values
            current_returns = current_returns[~np.isnan(current_returns)]
            
            if len(current_returns) > 0:
                # Mahalanobis distance
                try:
                    diff = current_returns - returns_pivot.mean().values[:len(current_returns)]
                    turbulence = np.dot(np.dot(diff.T, np.linalg.pinv(cov_matrix)), diff)
                except:
                    turbulence = 0
            else:
                turbulence = 0
            
            turbulence_list.append({'Date': date, 'turbulence': turbulence})
        
        turbulence_df = pd.DataFrame(turbulence_list)
        df = df.merge(turbulence_df, on='Date', how='left')
        
        self.processed_data = df
        print("Turbulence index calculated successfully")
        return self.processed_data
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Returns:
            Tuple of (train_data, test_data)
        """
        train_start = pd.to_datetime(self.config['data']['train_start'])
        train_end = pd.to_datetime(self.config['data']['train_end'])
        test_start = pd.to_datetime(self.config['data']['test_start'])
        test_end = pd.to_datetime(self.config['data']['test_end'])
        
        df = self.processed_data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        train_data = df[(df['Date'] >= train_start) & (df['Date'] <= train_end)]
        test_data = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)]
        
        print(f"Train data: {train_data['Date'].min()} to {train_data['Date'].max()}")
        print(f"Test data: {test_data['Date'].min()} to {test_data['Date'].max()}")
        
        return train_data, test_data
    
    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute all data processing steps.
        
        Returns:
            Tuple of (train_data, test_data)
        """
        self.fetch_data()
        self.calculate_technical_indicators()
        self.add_turbulence_index()
        train_data, test_data = self.split_data()
        
        return train_data, test_data


if __name__ == "__main__":
    # Test the data processor
    import yaml
    
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    processor = DataProcessor(config)
    train_data, test_data = processor.process_all()
    
    print("\nData processing complete!")
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
