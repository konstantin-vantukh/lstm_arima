"""
CLI Entry Point for Hybrid LSTM-ARIMA Forecasting System

This module serves as the main command-line interface for the hybrid forecasting system.
It parses CLI arguments, orchestrates the complete forecasting workflow, and exports results.

Usage:
    python forecaster.py --input data/prices.csv --ticker BTC --horizon 10
    python forecaster.py --input data/prices.json --ticker ETH --horizon 30 --output results.csv
    python forecaster.py --input data/prices.csv --ticker BTC --horizon 10 --config config/params.yml
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
from src.output_manager import export_results, format_results_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def load_default_config() -> Dict[str, Any]:
    """
    Load default model configuration.
    
    Returns:
        dict: Default configuration parameters
    """
    default_config = {
        'arima': {
            'seasonal': False,
            'max_p': 5,
            'max_d': 2,
            'max_q': 5,
            'information_criterion': 'aic'
        },
        'lstm': {
            'hidden_layers': 1,
            'nodes': 10,
            'batch_size': 64,
            'epochs': 100,
            'dropout_rate': 0.4,
            'l2_regularization': 0.01,
            'window_size': 60,
            'optimizer': 'adam',
            'early_stopping_patience': 10
        },
        'validation': {
            'method': 'walk_forward',
            'test_size': 0.2
        }
    }
    return default_config


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration parameters
        
    Raises:
        FileNotFoundError: If config file does not exist
        ImportError: If PyYAML is not installed
        yaml.YAMLError: If YAML parsing fails
    """
    if not YAML_AVAILABLE:
        error_msg = "PyYAML module is required for configuration file support. Install with: pip install PyYAML"
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        error_msg = f"Configuration file not found: {config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded successfully")
        return config
    except yaml.YAMLError as e:
        error_msg = f"Failed to parse configuration file {config_path}: {str(e)}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Error reading configuration file {config_path}: {str(e)}"
        logger.error(error_msg)
        raise


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
    Execute complete hybrid forecasting workflow.
    
    Implements the main orchestration function from architecture Section 9.1.
    Executes complete hybrid workflow: preprocessing → ARIMA → LSTM → combination
    
    Args:
        data (pd.DataFrame): Cleaned cryptocurrency price time series
        forecast_horizon (int): Number of time steps to forecast
        config (dict, optional): Model configuration overrides. If None, uses defaults.
    
    Returns:
        dict: Results containing:
            - predictions (np.ndarray): Combined hybrid forecast values
            - arima_component (np.ndarray): Linear ARIMA component predictions
            - lstm_component (np.ndarray): Non-linear LSTM residual predictions
            - metrics (dict): Performance metrics with RMSE and MAE
            - model_params (dict): Fitted ARIMA (p,d,q) parameters
            - component_stats (dict): Component contribution statistics
            - timestamp (str): Generation timestamp
            
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
        window_size = lstm_config.get('window_size', 60)
        
        # Check if we have enough data
        if len(residuals_array) < window_size + 1:
            logger.warning(
                f"Insufficient data for window_size={window_size}. "
                f"Data length: {len(residuals_array)}"
            )
            window_size = max(5, len(residuals_array) // 3)
            logger.info(f"Adjusted window_size to {window_size}")
        
        # Create rolling windows
        X, y = create_rolling_windows(residuals_array, window_size=window_size)
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
        if len(residuals_array) >= window_size:
            last_window = residuals_array[-window_size:]
            future_residual_forecast = []
            
            current_window = last_window.copy().reshape(-1, 1)
            for step in range(forecast_horizon):
                # Predict next step
                prediction = lstm_model.predict(
                    current_window.reshape(1, window_size, 1), 
                    verbose=0
                )[0, 0]
                future_residual_forecast.append(prediction)
                
                # Update window for next step
                current_window = np.vstack([current_window[1:], [[prediction]]])
            
            lstm_forecast = np.array(future_residual_forecast)
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
        
        # ===== STEP 5: EVALUATION =====
        logger.info("\n" + "-"*80)
        logger.info("STEP 5: EVALUATION AND METRICS")
        logger.info("-"*80)
        
        # Calculate metrics on historical fitted values for validation
        # Compare ARIMA + LSTM residual predictions vs actual returns
        if len(lstm_residual_predictions) > 0:
            arima_fitted = arima_model.fittedvalues.values[:len(lstm_residual_predictions)]
            hybrid_fitted = arima_fitted + lstm_residual_predictions
            
            rmse = calculate_rmse(returns.values[:len(lstm_residual_predictions)], hybrid_fitted)
            mae = calculate_mae(returns.values[:len(lstm_residual_predictions)], hybrid_fitted)
            
            logger.info(f"Validation metrics on historical data:")
            logger.info(f"  RMSE: {rmse:.6f}")
            logger.info(f"  MAE: {mae:.6f}")
            
            metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'validation_samples': len(lstm_residual_predictions)
            }
        else:
            metrics = {
                'rmse': 0.0,
                'mae': 0.0,
                'validation_samples': 0
            }
        
        # ===== BUILD RESULT DICTIONARY =====
        logger.info("\n" + "-"*80)
        logger.info("BUILDING RESULTS DICTIONARY")
        logger.info("-"*80)
        
        result = {
            'predictions': hybrid_predictions,
            'arima_component': arima_aligned,
            'lstm_component': lstm_aligned,
            'metrics': metrics,
            'model_params': {
                'arima_order': optimal_order,
                'lstm_window_size': window_size,
                'lstm_nodes': lstm_config.get('nodes', 10),
                'lstm_hidden_layers': lstm_config.get('hidden_layers', 1)
            },
            'component_stats': component_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("\n" + "="*80)
        logger.info("HYBRID FORECAST WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return result
        
    except Exception as e:
        error_msg = f"Hybrid forecast workflow failed: {str(e)}"
        logger.error(error_msg)
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
        
        # Load configuration
        if args.config is not None:
            config = load_config_from_file(args.config)
        else:
            config = load_default_config()
        
        # Load data
        logger.info(f"\nLoading data from {args.input}")
        data = load_data(args.input)
        logger.info(f"Data loaded successfully: shape={data.shape}")
        
        # Run hybrid forecast
        logger.info("\nStarting hybrid forecasting workflow...")
        results = run_hybrid_forecast(data, forecast_horizon=args.horizon, config=config)
        
        # Add metadata to results
        results['ticker'] = args.ticker
        results['horizon'] = args.horizon
        results['input_file'] = args.input
        
        # Format and display summary
        summary = format_results_summary(results)
        print("\n" + summary)
        
        # Export results
        if args.output == 'stdout':
            logger.info("Output to stdout requested. Summary displayed above.")
        else:
            # Determine output format
            output_path = args.output
            if output_path is not None:
                format_ext = Path(output_path).suffix.lower()
                if format_ext == '.json':
                    output_format = 'json'
                else:
                    output_format = 'csv'
            else:
                output_format = 'csv'
            
            logger.info(f"\nExporting results to {output_format.upper()}...")
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
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return 1
        
    except ValueError as e:
        error_msg = f"Validation Error: {str(e)}"
        logger.error(error_msg)
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return 1
        
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
