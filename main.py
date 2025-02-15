#!/usr/bin/env python3

"""
CryptoSage: Main entry point for the cryptocurrency analysis tool
"""

from src.cryptosage import CryptoSage

def main():
    # Initialize analysis
    analysis = CryptoSage()
    
    # Fetch market data
    df_bitcoin = analysis.fetch_market_data("BTC-USD")
    df_eth = analysis.fetch_market_data("ETH-USD")
    df_nasdaq = analysis.fetch_market_data("^NDX")
    
    if all([df_bitcoin is not None, df_eth is not None, df_nasdaq is not None]):
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
    else:
        print("Error: Failed to fetch one or more data sources")

if __name__ == "__main__":
    main()
