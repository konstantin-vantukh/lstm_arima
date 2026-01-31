"""
CLI Entry Point for Hybrid LSTM-ARIMA Forecasting System

This module serves as the main command-line interface for the hybrid forecasting system.
It parses CLI arguments, orchestrates the complete forecasting workflow, and exports results.

Usage:
    python forecaster.py --input data/prices.csv --ticker BTC --horizon 10
    python forecaster.py --input data/prices.json --ticker ETH --horizon 30 --output results.csv
    python forecaster.py --input data/prices.csv --ticker BTC --horizon 10 --config config/params.yml

Error Handling & Logging: Section 10, Error Handling Strategy
"""

import argparse
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Optional YAML support for configuration files
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Import all required modules
from src.preprocessing import load_data, impute_missing, calculate_returns, reshape_for_lstm
from src.arima_engine import test_stationarity, find_optimal_params, fit_arima, extract_residuals
from src.lstm_engine import build_lstm_model, create_rolling_windows, train_lstm, predict_residuals
from src.hybrid_combiner import combine_predictions, get_component_details
from src.evaluation import calculate_rmse, calculate_mae, create_metrics_report
from src.price_converter import reconstruct_price_series, validate_reconstructed_prices
from src.output_manager import export_results, format_results_summary
from src.config_loader import load_config, validate_config, get_default_config, merge_config, ConfigurationError

# Import error handling and logging framework
from src.exceptions import *
from src.logger_config import configure_logging, get_logger, log_exception

# Configure logging for the forecaster module
configure_logging(log_level='INFO', log_file='output/run.log')
logger = get_logger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure CLI argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Hybrid LSTM-ARIMA Forecasting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with CSV input
  python forecaster.py --input data/btc_prices.csv --ticker BTC --horizon 10
  
  # With JSON input and custom output
  python forecaster.py --input data/eth_prices.json --ticker ETH --horizon 30 --output results/forecast.csv
  
  # With custom configuration
  python forecaster.py --input data/btc_prices.csv --ticker BTC --horizon 10 --config config/custom_params.yml
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input file (CSV or JSON) containing price data'
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Asset ticker symbol (e.g., BTC, ETH, GOOG)'
    )
    
    parser.add_argument(
        '--horizon',
        type=int,
        required=True,
        help='Forecast horizon - number of periods to forecast'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: output/forecast_YYYYMMDD_HHMMSS.csv). Set to "stdout" to print to console'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model configuration file (YAML format). Uses defaults if not specified'
    )
    
    return parser




def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate CLI arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        
    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If arguments are invalid
    """
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        error_msg = f"Input file not found: {args.input}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Validate file extension
    if input_path.suffix.lower() not in ['.csv', '.json']:
        error_msg = f"Input file must be CSV or JSON, got: {input_path.suffix}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate ticker
    if not isinstance(args.ticker, str) or len(args.ticker) == 0:
        error_msg = "Ticker must be a non-empty string"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate horizon
    if not isinstance(args.horizon, int) or args.horizon <= 0:
        error_msg = f"Horizon must be a positive integer, got: {args.horizon}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate config file if provided
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            error_msg = f"Configuration file not found: {args.config}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    logger.info("All CLI arguments validated successfully")


def run_hybrid_forecast(
    data: pd.DataFrame,
    forecast_horizon: int,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Execute complete hybrid forecasting workflow with dual-space output.
    
    Implements the main orchestration function from architecture Section 9.1 and 9.2.
    Executes complete hybrid workflow: preprocessing → ARIMA → LSTM → combination → price reconstruction
    Generates both returns-space and price-space forecasts with metrics in both spaces.
    
    Per Architecture Section 4.1 (Pipeline):
        Input Data (CSV/JSON) → CLI Validation → Preprocess (Returns Space) → ARIMA Fitting
        → Extract Residuals → LSTM Training → Generate Forecasts → Hybrid Combination
        → Price Reconstruction → Calculate Metrics (Both Spaces) → Export Results
    
    Args:
        data (pd.DataFrame): Cleaned cryptocurrency price time series
        forecast_horizon (int): Number of time steps to forecast
        config (dict, optional): Model configuration overrides. If None, uses defaults.
    
    Returns:
        dict: Results containing (per Architecture Section 9.1 Interface Contract):
            - predictions_returns (np.ndarray): Combined hybrid forecast values (returns space)
            - predictions_price (np.ndarray): Reconstructed price forecast (price space)
            - arima_component (np.ndarray): Linear ARIMA component predictions (returns space)
            - lstm_component (np.ndarray): Non-linear LSTM residual predictions (returns space)
            - metrics_returns (dict): Performance metrics with RMSE and MAE (returns space)
            - metrics_price (dict): Performance metrics with RMSE and MAE (price space)
            - model_params (dict): Fitted ARIMA (p,d,q) parameters and last_price for reconstruction
            - component_stats (dict): Component contribution statistics
            - timestamp (str): Generation timestamp
            - last_price (float): Last observed price used for price reconstruction
            
    Raises:
        ValueError: If data is invalid or workflow fails
        Exception: For any processing errors
    """
    try:
        logger.info("="*80)
        logger.info("STARTING HYBRID FORECAST WORKFLOW")
        logger.info("="*80)
        
        # Load configuration
        if config is None:
            config = load_default_config()
            logger.info("Using default configuration")
        else:
            logger.info("Using provided configuration")
        
        # Extract configuration sections
        arima_config = config.get('arima', {})
        lstm_config = config.get('lstm', {})

        # Ensure that the LSTM config includes the explicit forecast_horizon
        # for the Dense layer in build_lstm_model in src/lstm_engine.py.
        lstm_config['horizon'] = forecast_horizon
        
        # ===== STEP 1: DATA PREPROCESSING =====
        logger.info("\n" + "-"*80)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("-"*80)
        
        # Assuming data has a 'close' price column (common in OHLCV data)
        if 'close' in data.columns:
            prices = data['close']
        elif 'Close' in data.columns:
            prices = data['Close']
        elif len(data.columns) > 0:
            # Use first column if no 'close' column
            prices = data.iloc[:, 0]
        else:
            raise ValueError("Data must contain price information")
        
        logger.info(f"Extracted price series: {len(prices)} data points")
        
        # Impute missing values
        prices = impute_missing(prices)
        
        # Calculate returns (log returns or simple returns)
        returns = calculate_returns(prices)
        returns = impute_missing(returns)  # Handle any NaN from first value
        
        logger.info(f"Returns calculated: mean={returns.mean():.6f}, std={returns.std():.6f}")
        
        # ===== STEP 2: ARIMA PROCESSING =====
        logger.info("\n" + "-"*80)
        logger.info("STEP 2: ARIMA LINEAR COMPONENT")
        logger.info("-"*80)
        
        # Test stationarity
        is_stationary, p_value = test_stationarity(returns)
        logger.info(f"Stationarity test: stationary={is_stationary}, p-value={p_value:.6f}")
        
        # Find optimal ARIMA parameters
        max_p = arima_config.get('max_p', 5)
        max_d = arima_config.get('max_d', 2)
        max_q = arima_config.get('max_q', 5)
        
        optimal_order = find_optimal_params(returns, max_p=max_p, max_d=max_d, max_q=max_q)
        logger.info(f"Optimal ARIMA order: {optimal_order}")
        
        # Fit ARIMA model
        arima_model = fit_arima(returns, order=optimal_order)
        
        # Extract residuals for LSTM
        residuals = extract_residuals(returns, arima_model)
        logger.info(f"Residuals extracted: mean={residuals.mean():.6f}, std={residuals.std():.6f}")
        
        # Generate ARIMA forecast
        arima_forecast = arima_model.get_forecast(steps=forecast_horizon)
        arima_predictions = arima_forecast.predicted_mean.values
        logger.info(f"ARIMA forecast generated: {len(arima_predictions)} period(s)")
        
        # ===== STEP 3: LSTM PROCESSING =====
        logger.info("\n" + "-"*80)
        logger.info("STEP 3: LSTM NON-LINEAR COMPONENT")
        logger.info("-"*80)
        
        # Prepare residuals for LSTM
        residuals_array = residuals.values
        window_size = lstm_config.get('window_size', 350)
        stride = lstm_config.get('stride', 175)
        
        # Check if we have enough data
        if len(residuals_array) < window_size + 1:
            logger.warning(
                f"Insufficient data for window_size={window_size}. "
                f"Data length: {len(residuals_array)}"
            )
            window_size = max(5, len(residuals_array) // 3)
            stride = max(1, window_size // 2)
            logger.info(f"Adjusted window_size to {window_size}")
        
        # Create parameters for rolling windows
        window_config = dict()
        window_config['window_size'] = window_size
        window_config['stride'] = stride
        window_config['horizon'] = forecast_horizon
        # Create rolling windows
        X, y = create_rolling_windows(data=residuals_array, config=window_config)
        logger.info(f"Rolling windows created: X shape={X.shape}, y shape={y.shape}")
        
        # Build LSTM model
        input_shape = (window_size, 1)
        lstm_model = build_lstm_model(lstm_config, input_shape=input_shape)
        logger.info("LSTM model built successfully")
        
        # Train LSTM
        lstm_model = train_lstm(lstm_model, X, y, lstm_config)
        logger.info("LSTM model training completed")
        
        # Predict residuals for all historical data
        lstm_residual_predictions = predict_residuals(lstm_model, X)
        logger.info(f"LSTM residual predictions: {len(lstm_residual_predictions)} values")
        
        # Generate forecast for future periods
        # Use last window of residuals to predict future residuals
        # Generate forecast for future periods
        # Use last window of residuals to predict future residuals directly
        if len(residuals_array) >= window_size:
            last_window = residuals_array[-window_size:]
            # Reshape for LSTM model input: (1, window_size, 1)
            input_for_prediction = last_window.reshape(1, window_size, 1) 

            # Predict entire forecast horizon at once
            lstm_forecast = lstm_model.predict(input_for_prediction, verbose=0)[0]
        else:
            lstm_forecast = np.zeros(forecast_horizon)
        
        logger.info(f"LSTM forecast generated: {len(lstm_forecast)} period(s)")
        
        # ===== STEP 4: HYBRID COMBINATION =====
        logger.info("\n" + "-"*80)
        logger.info("STEP 4: HYBRID COMBINATION")
        logger.info("-"*80)
        
        # Combine ARIMA and LSTM forecasts
        combined_result = combine_predictions(arima_predictions, lstm_forecast)
        
        hybrid_predictions = combined_result['predictions']
        arima_aligned = combined_result['arima_component']
        lstm_aligned = combined_result['lstm_component']
        component_stats = combined_result['component_stats']
        
        logger.info(f"Hybrid predictions generated: {len(hybrid_predictions)} values")
        logger.info(
            f"Component contributions: ARIMA={component_stats['arima_contribution']:.2f}%, "
            f"LSTM={component_stats['lstm_contribution']:.2f}%"
        )
        
        # ===== STEP 5: PRICE RECONSTRUCTION =====
        logger.info("\n" + "-"*80)
        logger.info("STEP 5: PRICE RECONSTRUCTION (RETURNS → PRICE SPACE)")
        logger.info("-"*80)
        
        # Get last price for reconstruction (from price series)
        last_price = float(prices.iloc[-1])
        logger.info(f"Last observed price: ${last_price:.2f}")
        
        # Reconstruct prices from hybrid returns forecast (Formula: P̂_t = P_{t-1} × (1 + R̂_t))
        try:
            predicted_prices = reconstruct_price_series(last_price, hybrid_predictions)
            logger.info(f"Price reconstruction successful")
            logger.info(f"  Price range: ${predicted_prices.min():.2f} - ${predicted_prices.max():.2f}")
            
            # Validate reconstructed prices
            price_validation = validate_reconstructed_prices(predicted_prices)
            if not price_validation['is_valid']:
                logger.warning(f"Price validation warnings: {price_validation['warnings']}")
            
        except Exception as e:
            logger.error(f"Price reconstruction failed: {str(e)}")
            # Fallback: keep predictions as-is if reconstruction fails
            predicted_prices = hybrid_predictions.copy()
            logger.warning("Falling back to returns-space predictions for price output")
        
        # ===== STEP 6: EVALUATION (DUAL-SPACE METRICS) =====
        logger.info("\n" + "-"*80)
        logger.info("STEP 6: EVALUATION AND METRICS (DUAL-SPACE)")
        logger.info("-"*80)
        
        # Calculate metrics on historical fitted values for validation in RETURNS SPACE
        metrics_returns = {}
        metrics_price = {}
        
        if len(lstm_residual_predictions) > 0:
            # Extract historical prices for comparison
            historical_prices = prices.values[:len(lstm_residual_predictions)+1]
            
            # Returns space metrics (compare actual vs fitted returns)
            arima_fitted = arima_model.fittedvalues.values[:len(lstm_residual_predictions)]
            hybrid_fitted = arima_fitted + lstm_residual_predictions[:, 0]
            actual_returns = returns.values[:len(lstm_residual_predictions)]
            
            rmse_returns = calculate_rmse(actual_returns, hybrid_fitted)
            mae_returns = calculate_mae(actual_returns, hybrid_fitted)
            
            metrics_returns = {
                'rmse': float(rmse_returns),
                'mae': float(mae_returns),
                'validation_samples': len(lstm_residual_predictions)
            }
            
            logger.info(f"Returns Space Metrics (Historical Validation):")
            logger.info(f"  RMSE: {rmse_returns:.6f}")
            logger.info(f"  MAE: {mae_returns:.6f}")
            
            # Price space metrics - reconstruct historical prices and compare
            try:
                reconstructed_historical = reconstruct_price_series(
                    historical_prices[0], 
                    hybrid_fitted
                )
                actual_historical_prices = historical_prices[1:len(hybrid_fitted)+1]
                
                rmse_price = calculate_rmse(actual_historical_prices, reconstructed_historical)
                mae_price = calculate_mae(actual_historical_prices, reconstructed_historical)
                
                metrics_price = {
                    'rmse': float(rmse_price),
                    'mae': float(mae_price),
                    'validation_samples': len(lstm_residual_predictions)
                }
                
                logger.info(f"Price Space Metrics (Historical Validation):")
                logger.info(f"  RMSE: ${rmse_price:.2f}")
                logger.info(f"  MAE: ${mae_price:.2f}")
                
            except Exception as e:
                logger.warning(f"Price space metrics calculation failed: {str(e)}")
                # Use empty dict if price space metrics cannot be calculated
                metrics_price = {
                    'rmse': 0.0,
                    'mae': 0.0,
                    'validation_samples': 0,
                    'error': str(e)
                }
        else:
            metrics_returns = {
                'rmse': 0.0,
                'mae': 0.0,
                'validation_samples': 0
            }
            metrics_price = {
                'rmse': 0.0,
                'mae': 0.0,
                'validation_samples': 0
            }
        
        # ===== BUILD RESULT DICTIONARY (DUAL-SPACE) =====
        logger.info("\n" + "-"*80)
        logger.info("BUILDING RESULTS DICTIONARY (DUAL-SPACE OUTPUT)")
        logger.info("-"*80)
        
        result = {
            # Dual-space predictions (per Architecture 9.1 interface contract)
            'predictions_returns': hybrid_predictions,      # Returns space forecast
            'predictions_price': predicted_prices,          # Price space forecast (reconstructed)
            'predictions': predicted_prices,                # For backward compatibility with output_manager
            'arima_component': arima_aligned,               # ARIMA linear (returns space)
            'lstm_component': lstm_aligned,                 # LSTM non-linear (returns space)
            # Dual-space metrics (per Architecture 9.1 interface contract)
            'metrics_returns': metrics_returns,             # Metrics in returns space
            'metrics_price': metrics_price,                 # Metrics in price space
            'metrics': {                                    # For backward compatibility
                **metrics_returns,
                'metrics_price': metrics_price
            },
            # Model parameters with last price for reconstruction
            'model_params': {
                'arima_order': optimal_order,
                'lstm_window_size': window_size,
                'lstm_nodes': lstm_config.get('nodes', 10),
                'lstm_hidden_layers': lstm_config.get('hidden_layers', 1),
                'last_price': float(last_price)  # For reference/reconstruction verification
            },
            # Component statistics
            'component_stats': component_stats,
            # Metadata
            'timestamp': datetime.now().isoformat(),
            'last_price': float(last_price)  # Exposed for easy access
        }
        
        logger.info("\n" + "="*80)
        logger.info("HYBRID FORECAST WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return result
        
    except Exception as e:
        error_msg = f"Hybrid forecast workflow failed: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise


def main():
    """
    Main entry point for the CLI application.
    
    Parses arguments, validates inputs, loads data, runs hybrid forecast,
    and exports results.
    """
    try:
        # Parse arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        logger.info("Command-line arguments parsed")
        logger.info(f"  Input: {args.input}")
        logger.info(f"  Ticker: {args.ticker}")
        logger.info(f"  Horizon: {args.horizon}")
        logger.info(f"  Output: {args.output if args.output else 'default'}")
        logger.info(f"  Config: {args.config if args.config else 'default'}")
        
        # Validate arguments
        validate_arguments(args)
        
        # Load configuration using new config_loader module
        try:
            if args.config is not None:
                config = load_config(config_path=args.config)
                logger.info(f"Loaded configuration from {args.config}")
            else:
                config = load_config()
                logger.info("Loaded default configuration")
        except ConfigurationError as e:
            error_msg = f"Configuration Error: {str(e)}"
            logger.error(error_msg)
            raise
        
        # Load data
        logger.info(f"\nLoading data from {args.input}")
        data = load_data(args.input)
        logger.info(f"Data loaded successfully: shape={data.shape}")
        
        # Run hybrid forecast
        logger.info("\nStarting hybrid forecasting workflow...")
        results = run_hybrid_forecast(data, forecast_horizon=args.horizon, config=config)
        
        # Add metadata to results (per Architecture 9.2)
        results['ticker'] = args.ticker
        results['horizon'] = args.horizon
        results['input_file'] = args.input
        
        # Format and display summary
        summary = format_results_summary(results)
        print("\n" + summary)
        
        # Export results with dual-space output
        if args.output == 'stdout':
            logger.info("Output to stdout requested. Summary displayed above.")
        else:
            # Determine output format and export dual-space results (per Architecture 9.2)
            output_path = args.output
            if output_path is not None:
                format_ext = Path(output_path).suffix.lower()
                if format_ext == '.json':
                    output_format = 'json'
                else:
                    output_format = 'csv'
            else:
                output_format = 'csv'
            
            logger.info(f"\nExporting dual-space results to {output_format.upper()}...")
            
            # For dual-space CSV export, use export_to_csv from output_manager
            if output_format == 'csv' and output_path:
                from src.output_manager import export_to_csv
                try:
                    export_to_csv(
                        output_path,
                        predictions_returns=results.get('predictions_returns'),
                        predictions_price=results.get('predictions_price'),
                        arima_component=results.get('arima_component'),
                        lstm_component=results.get('lstm_component'),
                        metrics_returns=results.get('metrics_returns'),
                        metrics_price=results.get('metrics_price')
                    )
                    logger.info(f"Results exported successfully to: {output_path}")
                    print(f"\n✓ Results saved to: {output_path}")
                except Exception as e:
                    logger.error(f"CSV export failed: {str(e)}, falling back to legacy export")
                    exported_path = export_results(results, output_path=output_path, format=output_format)
                    print(f"\n✓ Results saved to: {exported_path}")
            
            # For dual-space JSON export, use export_to_json from output_manager
            elif output_format == 'json' and output_path:
                from src.output_manager import export_to_json
                try:
                    export_to_json(
                        output_path,
                        ticker=args.ticker,
                        horizon=args.horizon,
                        predictions_returns=results.get('predictions_returns'),
                        predictions_price=results.get('predictions_price'),
                        arima_component=results.get('arima_component'),
                        lstm_component=results.get('lstm_component'),
                        metrics_returns=results.get('metrics_returns'),
                        metrics_price=results.get('metrics_price'),
                        model_params=results.get('model_params')
                    )
                    logger.info(f"Results exported successfully to: {output_path}")
                    print(f"\n✓ Results saved to: {output_path}")
                except Exception as e:
                    logger.error(f"JSON export failed: {str(e)}, falling back to legacy export")
                    exported_path = export_results(results, output_path=output_path, format=output_format)
                    print(f"\n✓ Results saved to: {exported_path}")
            else:
                # Fallback to legacy export
                exported_path = export_results(results, output_path=output_path, format=output_format)
                logger.info(f"Results exported successfully to: {exported_path}")
                print(f"\n✓ Results saved to: {exported_path}")
        
        logger.info("\n" + "="*80)
        logger.info("FORECASTING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return 0
        
    except FileNotFoundError as e:
        error_msg = f"File Error: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return 1
        
    except ValueError as e:
        error_msg = f"Validation Error: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return 1
        
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
