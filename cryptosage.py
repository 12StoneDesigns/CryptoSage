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
import pandas_ta as ta

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
        plt.style.use('seaborn')
        sns.set_palette('husl')
        
    def fetch_market_data(self, symbol):
        """Fetch market data for given symbol with error handling"""
        try:
            df = yf.download(symbol, start=self.start_date, end=date.today(), progress=False)
            print(f"Successfully fetched data for {symbol}")
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_custom_indicators(self, df, symbol_suffix):
        """Calculate technical indicators with custom parameters"""
        # RSI with multiple timeframes
        df[f"RSI(2)({symbol_suffix})"] = ta.rsi(close=df["Close"], length=2)
        df[f"RSI(10)({symbol_suffix})"] = ta.rsi(close=df["Close"], length=10)
        
        # Moving Average Ratios
        df[f"Close/MovingAvg(16)({symbol_suffix})"] = df["Close"] / ta.sma(close=df["Close"], length=16)
        df[f"Close/MovingAvg(35)({symbol_suffix})"] = df["Close"] / ta.sma(close=df["Close"], length=35)
        
        # Williams %R
        df[f"Williams %R(10)({symbol_suffix})"] = ta.willr(high=df["High"], low=df["Low"], 
                                                          close=df["Close"], lbp=10)
        
        # Custom Momentum Indicator
        df[f"Momentum({symbol_suffix})"] = df["Close"] / df["Close"].shift(10) * 100
        
        return df.dropna()

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

    def train_arima_model(self, train_data, test_data, order=(5,1,0)):
        """Train and evaluate ARIMA model"""
        model = ARIMA(train_data.astype(float), order=order)
        fitted_model = model.fit()
        
        predictions = fitted_model.forecast(len(test_data))
        rmse = np.sqrt(mean_squared_error(test_data.astype(float), predictions))
        
        return predictions, rmse

    def plot_correlation_matrix(self, df, title):
        """Plot correlation matrix"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(title)
        plt.show()

    def plot_predictions(self, actual, predictions, title):
        """Visualize predictions"""
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual.values.astype(float), 
                label="Actual", marker="o", alpha=0.6)
        plt.plot(actual.index, predictions, 
                label="Predicted", marker="x", alpha=0.8)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price Direction")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def main():
    # Initialize analysis
    analysis = CryptoSage()
    
    # Fetch market data
    df_bitcoin = analysis.fetch_market_data("BTC-USD")
    df_eth = analysis.fetch_market_data("ETH-USD")
    df_nasdaq = analysis.fetch_market_data("^NDX")
    
    # Calculate indicators
    df_bitcoin = analysis.calculate_custom_indicators(df_bitcoin, "BTC")
    df_eth = analysis.calculate_custom_indicators(df_eth, "ETH")
    df_nasdaq = analysis.calculate_custom_indicators(df_nasdaq, "NDX")
    
    # Plot correlation matrix for Bitcoin
    analysis.plot_correlation_matrix(df_bitcoin, 'Bitcoin Correlation Matrix')
    
    # Plot Williams %R for each asset
    analysis.plot_williams_r(df_bitcoin, "BTC")
    analysis.plot_williams_r(df_eth, "ETH")
    analysis.plot_williams_r(df_nasdaq, "NDX")
    
    # Prepare Bitcoin prediction data and train RIPPER model
    X_btc_train, X_btc_test, y_btc_train, y_btc_test = analysis.prepare_prediction_data(df_bitcoin)
    ripper_model = analysis.train_ripper_model(X_btc_train, y_btc_train)
    print("\nRIPPER Classification Rules:")
    ripper_model.out_model()
    
    # Prepare Ethereum prediction data and train ARIMA model
    X_eth_train, X_eth_test, y_eth_train, y_eth_test = analysis.prepare_prediction_data(df_eth)
    eth_predictions, eth_rmse = analysis.train_arima_model(y_eth_train, y_eth_test)
    print(f"\nEthereum ARIMA Model RMSE: {eth_rmse:.4f}")
    
    # Plot Ethereum predictions
    analysis.plot_predictions(y_eth_test, eth_predictions, 
                            "Ethereum - Actual vs Predicted Price Direction")

if __name__ == "__main__":
    main()
