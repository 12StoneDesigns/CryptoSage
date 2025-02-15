#!/usr/bin/env python3

"""
CryptoSage: Advanced Cryptocurrency Analysis with Machine Learning
Author: T. Landon Love
12 Stone Designs (12stonedesigns@gmail.com)
"""

import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import wittgenstein as wnt

# Time Series Analysis
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

class CryptoSage:
    def __init__(self, start_date="2019-01-01"):
        """Initialize the analysis with a start date"""
        self.start_date = start_date
        # Set style for all visualizations
        plt.style.use('seaborn-v0_8-darkgrid')  # Using a built-in style
        sns.set_theme(style="darkgrid")  # Set seaborn theme
        
    def fetch_market_data(self, symbol):
        """Fetch market data for given symbol with error handling"""
        try:
            df = yf.download(symbol, start=self.start_date, end=date.today(), progress=False)
            print(f"Successfully fetched data for {symbol}")
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_rsi(self, prices, periods=14):
        """Calculate RSI using pandas operations"""
        # Calculate price changes
        delta = prices.diff()
        
        # Create two series: gains (positive changes) and losses (negative changes)
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=periods).mean()
        avg_losses = losses.rolling(window=periods).mean()
        
        # Calculate relative strength
        rs = avg_gains / avg_losses
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def calculate_williams_r(self, high, low, close, periods=14):
        """Calculate Williams %R using pandas operations"""
        # Ensure inputs are pandas Series
        if not isinstance(high, pd.Series):
            high = pd.Series(high.squeeze())
        if not isinstance(low, pd.Series):
            low = pd.Series(low.squeeze())
        if not isinstance(close, pd.Series):
            close = pd.Series(close.squeeze())
            
        highest_high = high.rolling(window=periods).max()
        lowest_low = low.rolling(window=periods).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr

    def calculate_custom_indicators(self, df, symbol_suffix):
        """Calculate technical indicators with custom parameters"""
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Ensure the DataFrame is not empty
        if df.empty:
            raise ValueError("DataFrame is empty. Cannot calculate indicators.")
        
        # Check the shape of the DataFrame
        print(f"DataFrame shape: {df.shape}")
        
        try:
            # RSI with multiple timeframes
            df[f"RSI(2)({symbol_suffix})"] = self.calculate_rsi(df["Close"], periods=2)
            df[f"RSI(10)({symbol_suffix})"] = self.calculate_rsi(df["Close"], periods=10)
            
            # Moving Average Ratios
            df[f"Close/MovingAvg(16)({symbol_suffix})"] = df["Close"] / df["Close"].rolling(window=16).mean()
            df[f"Close/MovingAvg(35)({symbol_suffix})"] = df["Close"] / df["Close"].rolling(window=35).mean()
            
            # Williams %R
            df[f"Williams %R(10)({symbol_suffix})"] = self.calculate_williams_r(
                df["High"], df["Low"], df["Close"], periods=10
            )
            
            # Custom Momentum Indicator
            df[f"Momentum({symbol_suffix})"] = df["Close"] / df["Close"].shift(10) * 100
            
            return df.dropna()
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            raise

    def plot_market_data(self, df, title, columns):
        """Create enhanced visualizations for market data"""
        fig, axes = plt.subplots(nrows=len(columns), figsize=(12, 6*len(columns)))
        fig.suptitle(title, fontsize=16, y=1.02)
        
        for idx, column in enumerate(columns):
            df[column].plot(ax=axes[idx], title=column)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlabel('Date')
        
        plt.tight_layout()
        plt.show()

    def plot_williams_r(self, df, symbol):
        """Enhanced Williams %R visualization"""
        plt.figure(figsize=(12, 6))
        plt.plot(df[f"Williams %R(10)({symbol})"], label=f"Williams %R(10)({symbol})")
        plt.axhline(y=-20, color='r', linestyle='--', alpha=0.5, label='Overbought')
        plt.axhline(y=-80, color='g', linestyle='--', alpha=0.5, label='Oversold')
        plt.title(f"Williams %R (10 periods) - {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Williams %R")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def prepare_prediction_data(self, df, threshold=1.0025):
        """Prepare data for price direction prediction"""
        df["LABEL"] = np.where((df["Open"].shift(-2) / df["Open"].shift(-1)).gt(threshold), "1", "-1")
        df = df.dropna()
        
        X = df.drop(labels="LABEL", axis=1)
        y = df["LABEL"]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_ripper_model(self, X_train, y_train):
        """Train RIPPER classifier"""
        ripper = wnt.RIPPER(max_rules=100)
        ripper.fit(X_train, y_train, pos_class="1")
        return ripper

    def train_arima_model(self, train_data, test_data, order=(1,1,1)):
        """Train and evaluate ARIMA model"""
        try:
            # Ensure data is numeric and handle any missing values
            train_data = pd.to_numeric(train_data, errors='coerce').ffill()
            test_data = pd.to_numeric(test_data, errors='coerce').ffill()
            
            # Sort indices to ensure monotonic
            train_data = train_data.sort_index()
            test_data = test_data.sort_index()
            
            # Resample to daily frequency to ensure proper time series
            train_data = train_data.resample('D').ffill()
            test_data = test_data.resample('D').ffill()
            
            # Fit ARIMA model with simplified order
            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()
            
            # Generate predictions
            predictions = fitted_model.forecast(len(test_data))
            rmse = np.sqrt(mean_squared_error(test_data.values, predictions))
            
            return predictions, rmse
            
        except Exception as e:
            print(f"Error in ARIMA model: {str(e)}")
            # Return dummy predictions and high RMSE to indicate failure
            return np.zeros(len(test_data)), 999.99

    def plot_correlation_matrix(self, df, title):
        """Plot correlation matrix"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(title)
        plt.show()

    def plot_predictions(self, actual, predictions, title):
        """Visualize predictions"""
        plt.figure(figsize=(12, 6))
        
        # Ensure predictions have the same index as actual data
        predictions = pd.Series(predictions, index=actual.index[:len(predictions)])
        actual = actual[:len(predictions)]  # Trim actual data to match predictions
        
        plt.plot(actual.index, actual.values.astype(float), 
                label="Actual", marker="o", alpha=0.6)
        plt.plot(predictions.index, predictions.values, 
                label="Predicted", marker="x", alpha=0.8)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price Direction")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
