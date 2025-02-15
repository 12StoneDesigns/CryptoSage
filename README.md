# CryptoSage

Advanced Cryptocurrency Analysis with Machine Learning

## Overview

CryptoSage is a Python-based cryptocurrency analysis tool that combines technical analysis with machine learning to provide insights into cryptocurrency market trends. It features:

- Real-time market data fetching
- Custom technical indicators
- Machine learning-based price direction prediction
- Time series analysis using ARIMA models
- Advanced data visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/12StoneDesigns/CryptoSage.git
cd CryptoSage
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis script:
```bash
python main.py
```

This will:
- Fetch current market data for Bitcoin, Ethereum, and NASDAQ
- Calculate technical indicators
- Generate correlation matrices and visualizations
- Train and evaluate machine learning models
- Display price predictions

## Project Structure

```
CryptoSage/
├── data/               # Data storage directory
├── docs/              # Documentation
├── src/               # Source code
│   └── cryptosage/    
│       ├── __init__.py
│       └── analyzer.py
├── tests/             # Test files
├── main.py            # Main entry point
├── README.md          # Project documentation
└── requirements.txt   # Project dependencies
```

## Features

- **Market Data Analysis**
  - Real-time data fetching using yfinance
  - Custom technical indicators including RSI, Moving Averages, and Williams %R

- **Machine Learning Models**
  - RIPPER algorithm for price direction classification
  - ARIMA models for time series prediction

- **Visualization**
  - Correlation matrices
  - Technical indicator plots
  - Price prediction visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request to https://github.com/12StoneDesigns/CryptoSage.

## Author

T. Landon Love  
12 Stone Designs (12stonedesigns@gmail.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
