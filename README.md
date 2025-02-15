# CryptoSage: Advanced Cryptocurrency Analysis with Machine Learning

## Author
T. Landon Love  
12 Stone Designs (12stonedesigns@gmail.com)

## Description
CryptoSage is an advanced intermarket analysis system that explores the relationships between cryptocurrency markets (Bitcoin and Ethereum) and traditional markets (NASDAQ) using sophisticated technical indicators and machine learning approaches.

## Key Features
- Real-time data fetching using yfinance API
- Custom technical indicators including RSI, Williams %R, and Moving Averages
- RIPPER algorithm implementation for rule-based classification
- ARIMA time-series modeling for price direction prediction
- Enhanced visualization of market trends and indicators
- Comprehensive error handling and data validation

## Technical Implementation
- **Data Collection**: Historical price data for BTC-USD, ETH-USD, and NASDAQ (^NDX)
- **Feature Engineering**: Custom technical indicators using pandas_ta
- **Machine Learning**: Rule-based classifier (RIPPER) for Bitcoin price direction prediction
- **Time Series Analysis**: ARIMA modeling for Ethereum price forecasting
- **Performance Evaluation**: RMSE-based model assessment
- **Visualization**: Enhanced plots for technical indicators and predictions

## Dependencies
```
numpy
pandas
seaborn
yfinance
matplotlib
pandas_ta
scikit-learn
wittgenstein
statsmodels
```

## Installation
Install the required packages using pip:
```bash
pip install numpy pandas seaborn yfinance matplotlib pandas_ta scikit-learn wittgenstein statsmodels
```

## Usage
Run the analysis script:
```bash
python cryptosage.py
```

The script will:
1. Fetch market data for Bitcoin, Ethereum, and NASDAQ
2. Calculate technical indicators
3. Display correlation matrices and Williams %R indicators
4. Train and evaluate the RIPPER classifier for Bitcoin price prediction
5. Train and evaluate the ARIMA model for Ethereum price forecasting
6. Generate visualizations of the results

## Output
The script generates several visualizations:
- Correlation matrices for market relationships
- Williams %R indicators for each asset
- ARIMA model predictions vs actual values
- Technical indicator plots

It also outputs:
- RIPPER classification rules for Bitcoin price prediction
- ARIMA model RMSE for Ethereum price forecasting

## Note
CryptoSage provides insights into market relationships and potential trading opportunities through a combination of traditional technical analysis and modern machine learning approaches. The results should not be considered financial advice.
