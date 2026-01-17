"""
Output Manager Module for Hybrid LSTM-ARIMA Forecasting System

This module handles exporting results and reporting progress during the forecasting
pipeline. It provides functionality to export predictions and metrics to CSV/JSON formats
with dual-space output (returns and price space), log training progress to STDOUT, and
generate human-readable forecast summaries.

Key Features:
    - Dual-space output: Returns-space and price-space forecasts in all exports
    - CSV export with component breakdown
    - JSON export with structured metadata
    - STDOUT progress reporting
    - Comprehensive error handling
    - Path validation and directory creation

Functions:
    - export_to_csv: Export dual-space results to CSV
    - export_to_json: Export dual-space results to JSON
    - export_to_stdout: Print human-readable summary to console
    - log_progress: Log progress with structured logging
    - validate_output_path: Validate and create output directories
    - export_results: Main export function (legacy compatibility)
    - log_progress: (legacy compatibility)
    - format_results_summary: Create human-readable summary

Error Handling & Logging: Section 10, Error Handling Strategy
"""

import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np

from src.exceptions import FileIOError
from src.logger_config import get_logger, log_exception

# Configure module logging
logger = get_logger(__name__)


# ============================================================================
# DUAL-SPACE OUTPUT FUNCTIONS (ARCHITECTURE 9.2)
# ============================================================================

def validate_output_path(output_path: str) -> bool:
    """
    Validate and prepare output directory for file writing.
    
    Ensures the output directory exists and is writable. Creates parent 
    directories if needed using mkdir with parents=True.
    
    Args:
        output_path (str): Full file path where output will be written.
            Can be absolute or relative path.
    
    Returns:
        bool: True if directory is valid/created and writable, False otherwise.
    
    Raises:
        None - logs errors instead of raising
    
    Examples:
        >>> is_valid = validate_output_path("output/forecast.csv")
        >>> if is_valid:
        ...     print("Output path ready")
    """
    try:
        file_path = Path(output_path)
        parent_dir = file_path.parent
        
        # Create parent directories
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Check write permissions by testing with parent directory
        if not parent_dir.exists():
            logger.error(f"Failed to create output directory: {parent_dir}")
            return False
        
        # Verify write permission
        try:
            test_file = parent_dir / ".write_test_tmp"
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            logger.error(f"No write permission for directory {parent_dir}: {str(e)}")
            return False
        
        logger.info(f"Output path validated: {parent_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error validating output path {output_path}: {str(e)}")
        return False


def export_to_csv(
    output_path: str,
    predictions_returns: Union[np.ndarray, List],
    predictions_price: Union[np.ndarray, List],
    arima_component: Union[np.ndarray, List],
    lstm_component: Union[np.ndarray, List],
    metrics_returns: Optional[Dict[str, float]] = None,
    metrics_price: Optional[Dict[str, float]] = None
) -> None:
    """
    Export dual-space forecasts and components to CSV file.
    
    Creates a CSV file with columns for returns-space and price-space predictions,
    along with ARIMA and LSTM component breakdowns. Optionally includes metrics
    as summary rows at the end.
    
    CSV Format (from architecture 9.2):
        timestamp,prediction_returns,prediction_price,arima_component,lstm_component
        t1,0.0125,50125.50,0.0100,0.0025
        t2,0.0089,50570.25,0.0080,-0.0009
        t3,-0.0034,50296.18,-0.0045,0.0011
    
    Args:
        output_path (str): File path where CSV will be written.
        predictions_returns (np.ndarray or list): Forecast values in returns space.
        predictions_price (np.ndarray or list): Forecast values in price space.
        arima_component (np.ndarray or list): ARIMA linear predictions (returns space).
        lstm_component (np.ndarray or list): LSTM residual predictions (returns space).
        metrics_returns (dict, optional): Metrics in returns space with keys 'rmse', 'mae'.
        metrics_price (dict, optional): Metrics in price space with keys 'rmse', 'mae'.
    
    Returns:
        None
    
    Raises:
        IOError: If file write fails
        ValueError: If input arrays have mismatched lengths
    
    Examples:
        >>> import numpy as np
        >>> preds_ret = np.array([0.0125, 0.0089, -0.0034])
        >>> preds_price = np.array([50125.50, 50570.25, 50296.18])
        >>> arima = np.array([0.0100, 0.0080, -0.0045])
        >>> lstm = np.array([0.0025, -0.0009, 0.0011])
        >>> export_to_csv("output/forecast.csv", preds_ret, preds_price, arima, lstm)
    """
    try:
        # Validate inputs
        if not validate_output_path(output_path):
            raise IOError(f"Failed to validate output path: {output_path}")
        
        # Convert numpy arrays to lists
        if hasattr(predictions_returns, 'tolist'):
            predictions_returns = predictions_returns.tolist()
        if hasattr(predictions_price, 'tolist'):
            predictions_price = predictions_price.tolist()
        if hasattr(arima_component, 'tolist'):
            arima_component = arima_component.tolist()
        if hasattr(lstm_component, 'tolist'):
            lstm_component = lstm_component.tolist()
        
        # Validate array lengths match
        n_predictions = len(predictions_returns)
        if not (len(predictions_price) == n_predictions and 
                len(arima_component) == n_predictions and 
                len(lstm_component) == n_predictions):
            error_msg = (f"Input arrays have mismatched lengths: "
                        f"returns={len(predictions_returns)}, "
                        f"price={len(predictions_price)}, "
                        f"arima={len(arima_component)}, "
                        f"lstm={len(lstm_component)}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Write CSV file
        file_path = Path(output_path)
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'prediction_returns', 'prediction_price', 
                         'arima_component', 'lstm_component']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write data rows
            for i in range(n_predictions):
                row = {
                    'timestamp': f"t{i+1}",
                    'prediction_returns': predictions_returns[i],
                    'prediction_price': predictions_price[i],
                    'arima_component': arima_component[i],
                    'lstm_component': lstm_component[i]
                }
                writer.writerow(row)
            
            # Write metrics as summary rows if provided
            if metrics_returns or metrics_price:
                f.write("\n# Metrics Summary\n")
                
                if metrics_returns:
                    f.write("# Metrics (Returns Space)\n")
                    for metric_name, metric_value in metrics_returns.items():
                        f.write(f"# {metric_name}: {metric_value}\n")
                
                if metrics_price:
                    f.write("# Metrics (Price Space)\n")
                    for metric_name, metric_value in metrics_price.items():
                        f.write(f"# {metric_name}: {metric_value}\n")
        
        logger.info(f"CSV export successful: {file_path} ({n_predictions} predictions)")
    
    except (IOError, ValueError) as e:
        logger.error(f"Error in export_to_csv: {str(e)}")
        log_exception(logger, e)
        raise FileIOError(
            error_message=f"CSV export failed: {str(e)}",
            file_path=output_path,
            operation="write"
        ) from e
    except Exception as e:
        error_msg = f"Unexpected error during CSV export: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise FileIOError(
            error_message=error_msg,
            file_path=output_path,
            operation="write"
        ) from e


def export_to_json(
    output_path: str,
    ticker: str,
    horizon: int,
    predictions_returns: Union[np.ndarray, List],
    predictions_price: Union[np.ndarray, List],
    arima_component: Union[np.ndarray, List],
    lstm_component: Union[np.ndarray, List],
    metrics_returns: Optional[Dict[str, float]] = None,
    metrics_price: Optional[Dict[str, float]] = None,
    model_params: Optional[Dict[str, Any]] = None
) -> None:
    """
    Export dual-space forecasts and metadata to JSON file.
    
    Creates a JSON file with complete forecasting results including predictions
    in both spaces, components, metrics, and model parameters. Uses ISO 8601
    format for timestamps.
    
    JSON Format (from architecture 9.2):
    {
      "timestamp": "2026-01-15T08:00:00",
      "ticker": "BTC",
      "horizon": 10,
      "predictions_returns": [0.0125, 0.0089, -0.0034, ...],
      "predictions_price": [50125.50, 50570.25, 50296.18, ...],
      "arima_component": [0.0100, 0.0080, -0.0045, ...],
      "lstm_component": [0.0025, -0.0009, 0.0011, ...],
      "metrics_returns": {"rmse": 0.0045, "mae": 0.0032},
      "metrics_price": {"rmse": 225.50, "mae": 160.25},
      "model_params": {"arima_order": [1, 1, 1], "last_price": 50000.00}
    }
    
    Args:
        output_path (str): File path where JSON will be written.
        ticker (str): Asset ticker symbol (e.g., 'BTC', 'ETH').
        horizon (int): Forecast horizon (number of periods).
        predictions_returns (np.ndarray or list): Forecast values in returns space.
        predictions_price (np.ndarray or list): Forecast values in price space.
        arima_component (np.ndarray or list): ARIMA linear predictions (returns space).
        lstm_component (np.ndarray or list): LSTM residual predictions (returns space).
        metrics_returns (dict, optional): Metrics in returns space with 'rmse', 'mae'.
        metrics_price (dict, optional): Metrics in price space with 'rmse', 'mae'.
        model_params (dict, optional): Model parameters including 'arima_order', 'last_price'.
    
    Returns:
        None
    
    Raises:
        IOError: If file write fails
        ValueError: If input arrays have mismatched lengths or invalid ticker/horizon
    
    Examples:
        >>> import numpy as np
        >>> preds_ret = np.array([0.0125, 0.0089])
        >>> preds_price = np.array([50125.50, 50570.25])
        >>> arima = np.array([0.0100, 0.0080])
        >>> lstm = np.array([0.0025, -0.0009])
        >>> export_to_json("output/forecast.json", "BTC", 2, 
        ...                preds_ret, preds_price, arima, lstm)
    """
    try:
        # Validate inputs
        if not validate_output_path(output_path):
            raise IOError(f"Failed to validate output path: {output_path}")
        
        if not isinstance(ticker, str) or len(ticker) == 0:
            raise ValueError(f"ticker must be non-empty string, got {ticker}")
        
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"horizon must be positive integer, got {horizon}")
        
        # Convert numpy arrays to lists
        if hasattr(predictions_returns, 'tolist'):
            predictions_returns = predictions_returns.tolist()
        if hasattr(predictions_price, 'tolist'):
            predictions_price = predictions_price.tolist()
        if hasattr(arima_component, 'tolist'):
            arima_component = arima_component.tolist()
        if hasattr(lstm_component, 'tolist'):
            lstm_component = lstm_component.tolist()
        
        # Validate array lengths match
        n_predictions = len(predictions_returns)
        if not (len(predictions_price) == n_predictions and 
                len(arima_component) == n_predictions and 
                len(lstm_component) == n_predictions):
            error_msg = (f"Input arrays have mismatched lengths: "
                        f"returns={len(predictions_returns)}, "
                        f"price={len(predictions_price)}, "
                        f"arima={len(arima_component)}, "
                        f"lstm={len(lstm_component)}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Build JSON structure
        json_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ticker": ticker,
            "horizon": horizon,
            "predictions_returns": predictions_returns,
            "predictions_price": predictions_price,
            "arima_component": arima_component,
            "lstm_component": lstm_component
        }
        
        # Add optional metrics
        if metrics_returns:
            json_data["metrics_returns"] = metrics_returns
        if metrics_price:
            json_data["metrics_price"] = metrics_price
        
        # Add optional model parameters
        if model_params:
            json_data["model_params"] = model_params
        
        # Write JSON file
        file_path = Path(output_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, allow_nan=True)
        
        logger.info(f"JSON export successful: {file_path} ({n_predictions} predictions)")
    
    except (IOError, ValueError) as e:
        logger.error(f"Error in export_to_json: {str(e)}")
        log_exception(logger, e)
        raise FileIOError(
            error_message=f"JSON export failed: {str(e)}",
            file_path=output_path,
            operation="write"
        ) from e
    except Exception as e:
        error_msg = f"Unexpected error during JSON export: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise FileIOError(
            error_message=error_msg,
            file_path=output_path,
            operation="write"
        ) from e


def export_to_stdout(
    predictions_returns: Union[np.ndarray, List],
    predictions_price: Union[np.ndarray, List],
    metrics_returns: Optional[Dict[str, float]] = None,
    metrics_price: Optional[Dict[str, float]] = None
) -> None:
    """
    Print human-readable summary of dual-space forecasts to console.
    
    Displays first and last few predictions in both returns and price space,
    along with metrics summary if provided. Useful for progress reporting
    during forecasting.
    
    Args:
        predictions_returns (np.ndarray or list): Forecast values in returns space.
        predictions_price (np.ndarray or list): Forecast values in price space.
        metrics_returns (dict, optional): Metrics in returns space.
        metrics_price (dict, optional): Metrics in price space.
    
    Returns:
        None
    
    Examples:
        >>> import numpy as np
        >>> preds_ret = np.array([0.0125, 0.0089, -0.0034, 0.0045, 0.0067])
        >>> preds_price = np.array([50125.50, 50570.25, 50296.18, 50518.42, 50850.99])
        >>> export_to_stdout(preds_ret, preds_price)
        
        >>> # With metrics
        >>> metrics_r = {"rmse": 0.0045, "mae": 0.0032}
        >>> metrics_p = {"rmse": 225.50, "mae": 160.25}
        >>> export_to_stdout(preds_ret, preds_price, metrics_r, metrics_p)
    """
    try:
        # Convert numpy arrays to lists
        if hasattr(predictions_returns, 'tolist'):
            predictions_returns = predictions_returns.tolist()
        if hasattr(predictions_price, 'tolist'):
            predictions_price = predictions_price.tolist()
        
        n_predictions = len(predictions_returns)
        
        # Print header
        print("\n" + "=" * 80)
        print("HYBRID LSTM-ARIMA FORECAST - DUAL-SPACE PREDICTIONS")
        print("=" * 80)
        
        # Show first few predictions
        n_show = min(3, n_predictions)
        print(f"\nFirst {n_show} Predictions:")
        print("-" * 80)
        print(f"{'Index':<8} {'Returns':<18} {'Price':<18}")
        print("-" * 80)
        for i in range(n_show):
            print(f"t{i+1:<7} {predictions_returns[i]:<18.6f} {predictions_price[i]:<18.2f}")
        
        # Show last few predictions if different
        if n_predictions > n_show:
            start_idx = max(n_show, n_predictions - n_show)
            print(f"\nLast {n_predictions - start_idx} Predictions:")
            print("-" * 80)
            print(f"{'Index':<8} {'Returns':<18} {'Price':<18}")
            print("-" * 80)
            for i in range(start_idx, n_predictions):
                print(f"t{i+1:<7} {predictions_returns[i]:<18.6f} {predictions_price[i]:<18.2f}")
        
        # Print summary statistics
        print("\n" + "-" * 80)
        print("FORECAST SUMMARY")
        print("-" * 80)
        print(f"Total Predictions: {n_predictions}")
        print(f"Returns Range: [{min(predictions_returns):.6f}, {max(predictions_returns):.6f}]")
        print(f"Price Range: [{min(predictions_price):.2f}, {max(predictions_price):.2f}]")
        
        # Print metrics if provided
        if metrics_returns or metrics_price:
            print("\n" + "-" * 80)
            print("PERFORMANCE METRICS")
            print("-" * 80)
            
            if metrics_returns:
                print("Returns Space:")
                for metric_name, metric_value in metrics_returns.items():
                    print(f"  {metric_name.upper():<12}: {metric_value:.6f}")
            
            if metrics_price:
                print("Price Space:")
                for metric_name, metric_value in metrics_price.items():
                    print(f"  {metric_name.upper():<12}: {metric_value:.2f}")
        
        print("\n" + "=" * 80 + "\n")
        logger.info("STDOUT export completed")
    
    except Exception as e:
        logger.error(f"Error in export_to_stdout: {str(e)}")
        raise


def log_progress(message: str, level: str = 'INFO') -> None:
    """
    Log structured progress messages with configurable log level.
    
    Provides unified logging for progress reporting during model training and
    forecasting. Supports both INFO and WARNING levels. Uses Python's logging
    module for proper timestamp and source tracking.
    
    Args:
        message (str): Progress message to log.
        level (str): Log level - 'INFO' or 'WARNING'. Default: 'INFO'
    
    Returns:
        None
    
    Raises:
        ValueError: If level not in ['INFO', 'WARNING']
        TypeError: If message is not a string
    
    Examples:
        >>> log_progress("Starting ARIMA model fitting")
        >>> log_progress("GPU memory warning", level='WARNING')
        >>> log_progress("Hybrid model forecast complete")
    """
    try:
        # Validate inputs
        if not isinstance(message, str):
            raise TypeError(f"message must be string, got {type(message)}")
        
        if level not in ('INFO', 'WARNING'):
            raise ValueError(f"level must be 'INFO' or 'WARNING', got '{level}'")
        
        # Log with appropriate level
        if level == 'INFO':
            logger.info(message)
        elif level == 'WARNING':
            logger.warning(message)
    
    except (TypeError, ValueError) as e:
        logger.error(f"Error in log_progress: {str(e)}")
        raise


# ============================================================================
# LEGACY EXPORT FUNCTIONS (BACKWARD COMPATIBILITY)
# ============================================================================

def export_results(
    results: dict,
    output_path: Optional[str] = None,
    format: str = 'csv'
) -> str:
    """
    Export forecast results to CSV or JSON file.

    Exports comprehensive forecasting results including predictions, metrics,
    timestamps, and model parameters to the specified file format. Handles file I/O
    errors gracefully with informative error messages.

    Args:
        results (dict): Dictionary containing forecast results with keys:
            - predictions (required): np.ndarray or list of predicted values
            - arima_component (required): np.ndarray or list of ARIMA linear component
            - lstm_component (required): np.ndarray or list of LSTM non-linear component
            - metrics (optional): dict with RMSE, MAE and other performance metrics
            - model_params (optional): dict containing model configuration parameters
            - timestamps (optional): list of timestamps for each prediction
            - ticker (optional): asset ticker symbol
            - horizon (optional): forecast horizon in periods
        output_path (str, optional): Path where results file will be saved.
            If None, generates default path: output/forecast_YYYYMMDD_HHMMSS.{format}
            Path must be valid and writable. Parent directory must exist or will be created.
        format (str): Output file format, either 'csv' or 'json'. Default: 'csv'
            Raises ValueError if format is not one of these options.

    Returns:
        str: Absolute path to the exported results file as string.

    Raises:
        ValueError: If format is not 'csv' or 'json', or if results dict is missing
            required keys (predictions, arima_component, lstm_component).
        TypeError: If results is not a dictionary or other input types are invalid.
        FileNotFoundError: If output_path parent directory does not exist and cannot be created.
        PermissionError: If no write permission for the output directory.
        IOError: If file writing fails due to disk space or other I/O issues.

    Examples:
        >>> import numpy as np
        >>> results = {
        ...     'predictions': np.array([100.5, 101.2, 102.1]),
        ...     'arima_component': np.array([100.0, 101.0, 102.0]),
        ...     'lstm_component': np.array([0.5, 0.2, 0.1]),
        ...     'metrics': {'rmse': 0.35, 'mae': 0.27},
        ...     'model_params': {'arima_params': (1, 1, 1), 'lstm_nodes': 10}
        ... }
        >>> path = export_results(results, format='json')
        >>> print(f"Results exported to: {path}")

        >>> # With custom output path
        >>> path = export_results(results, output_path='results/forecast.csv', format='csv')
    """
    try:
        # Validate inputs
        if not isinstance(results, dict):
            error_msg = f"results must be a dictionary, got {type(results)}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Validate format parameter
        if format not in ('csv', 'json'):
            error_msg = f"format must be 'csv' or 'json', got '{format}'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate required keys in results
        required_keys = ['predictions', 'arima_component', 'lstm_component']
        missing_keys = [k for k in required_keys if k not in results]
        if missing_keys:
            error_msg = f"results missing required keys: {missing_keys}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Generate default output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path('output')
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"forecast_{timestamp}.{format}"
        else:
            file_path = Path(output_path)

        # Create parent directory if it doesn't exist
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            error_msg = f"Permission denied creating directory {file_path.parent}: {str(e)}"
            logger.error(error_msg)
            raise PermissionError(error_msg) from e
        except OSError as e:
            error_msg = f"Failed to create directory {file_path.parent}: {str(e)}"
            logger.error(error_msg)
            raise IOError(error_msg) from e

        logger.info(f"Exporting results to {format.upper()}: {file_path}")

        # Convert numpy arrays to lists for serialization
        export_data = {k: (v.tolist() if hasattr(v, 'tolist') else v)
                       for k, v in results.items()}

        if format == 'json':
            _export_to_json(export_data, file_path)
        else:  # format == 'csv'
            _export_to_csv(export_data, file_path)

        logger.info(f"Results successfully exported to: {file_path}")
        return str(file_path.absolute())

    except (ValueError, TypeError, PermissionError, IOError) as e:
        logger.error(f"Error in export_results: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Unexpected error during results export: {str(e)}"
        logger.error(error_msg)
        raise IOError(error_msg) from e


def _export_to_json(data: dict, file_path: Path) -> None:
    """
    Export data to JSON file.

    Helper function to handle JSON export with error handling.

    Args:
        data (dict): Data to export
        file_path (Path): Pathlib Path object for output file

    Raises:
        IOError: If JSON write fails
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, allow_nan=True)
        logger.debug(f"Successfully wrote JSON file: {file_path}")
    except (IOError, OSError) as e:
        error_msg = f"Failed to write JSON file {file_path}: {str(e)}"
        logger.error(error_msg)
        raise IOError(error_msg) from e
    except TypeError as e:
        error_msg = f"JSON serialization error for {file_path}: {str(e)}"
        logger.error(error_msg)
        raise IOError(error_msg) from e


def _export_to_csv(data: dict, file_path: Path) -> None:
    """
    Export data to CSV file.

    Helper function to handle CSV export with error handling.
    Converts nested structures to flattened rows.

    Args:
        data (dict): Data to export
        file_path (Path): Pathlib Path object for output file

    Raises:
        IOError: If CSV write fails
    """
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            # Extract prediction arrays
            predictions = data.get('predictions', [])
            arima_comp = data.get('arima_component', [])
            lstm_comp = data.get('lstm_component', [])
            timestamps = data.get('timestamps', [])

            # Determine number of rows
            n_rows = len(predictions) if isinstance(predictions, list) else 0

            # Pad timestamps if needed
            if not timestamps:
                timestamps = [f"t{i}" for i in range(n_rows)]
            elif len(timestamps) < n_rows:
                timestamps.extend([f"t{i}" for i in range(len(timestamps), n_rows)])

            # Write header
            fieldnames = ['timestamp', 'prediction', 'arima_component', 'lstm_component']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Write data rows
            for i in range(n_rows):
                row = {
                    'timestamp': timestamps[i] if i < len(timestamps) else f"t{i}",
                    'prediction': predictions[i] if i < len(predictions) else '',
                    'arima_component': arima_comp[i] if i < len(arima_comp) else '',
                    'lstm_component': lstm_comp[i] if i < len(lstm_comp) else ''
                }
                writer.writerow(row)

            # Write metrics and model params as comments
            if 'metrics' in data or 'model_params' in data:
                f.write('\n# Metadata\n')
                if 'metrics' in data:
                    f.write(f"# Metrics: {json.dumps(data['metrics'])}\n")
                if 'model_params' in data:
                    f.write(f"# Model Parameters: {json.dumps(data['model_params'])}\n")

        logger.debug(f"Successfully wrote CSV file: {file_path}")

    except (IOError, OSError) as e:
        error_msg = f"Failed to write CSV file {file_path}: {str(e)}"
        logger.error(error_msg)
        raise IOError(error_msg) from e


def log_epoch_progress(
    epoch: int,
    loss: float,
    stage: str = 'training'
) -> None:
    """
    Log training progress to STDOUT (Legacy compatibility function).

    Reports model training progress including epoch number, loss value, and training
    stage. Output goes to standard output stream for real-time monitoring of training.
    Loss values are formatted to 6 decimal places for readability.

    Args:
        epoch (int): Current epoch number, must be non-negative integer.
            Typically starts from 1 and increments each epoch.
        loss (float): Loss value for current epoch.
            Should be numeric (float, int) and typically non-negative for most loss functions.
        stage (str): Training stage identifier. Default: 'training'
            Examples: 'training', 'validation', 'test'
            Used to distinguish different phases of model training.

    Returns:
        None

    Raises:
        TypeError: If epoch is not an integer or loss is not numeric.
        ValueError: If epoch is negative.

    Examples:
        >>> log_epoch_progress(1, 0.5234567, stage='training')
        >>> # Output to STDOUT: [Epoch 1 | training] Loss: 0.523457

        >>> log_epoch_progress(10, 0.0987654, stage='validation')
        >>> # Output to STDOUT: [Epoch 10 | validation] Loss: 0.098765
    """
    try:
        # Validate epoch parameter
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            error_msg = f"epoch must be an integer, got {type(epoch)}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        if epoch < 0:
            error_msg = f"epoch must be non-negative, got {epoch}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate loss parameter
        try:
            loss_float = float(loss)
        except (TypeError, ValueError) as e:
            error_msg = f"loss must be numeric, got {type(loss)}"
            logger.error(error_msg)
            raise TypeError(error_msg) from e

        # Validate stage parameter
        if not isinstance(stage, str):
            error_msg = f"stage must be a string, got {type(stage)}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Format and print progress message directly to STDOUT
        progress_msg = f"[Epoch {epoch} | {stage}] Loss: {loss_float:.6f}"
        print(progress_msg, flush=True)

        # Also log for debugging purposes
        logger.info(f"Progress logged: {progress_msg}")

    except (TypeError, ValueError) as e:
        logger.error(f"Error in log_epoch_progress: {str(e)}")
        raise


def format_results_summary(results: dict) -> str:
    """
    Create human-readable summary of forecast results.

    Generates a formatted text summary of forecasting results including predictions,
    metrics, model parameters, and component statistics. Suitable for display to users
    or writing to reports.

    Args:
        results (dict): Dictionary containing forecast results with keys:
            - predictions (required): np.ndarray or list of predicted values
            - metrics (optional): dict with RMSE, MAE and other performance metrics
                Keys: 'rmse', 'mae', etc.
            - model_params (optional): dict containing model parameters
                Keys: 'arima_params', 'lstm_nodes', etc.
            - arima_component (optional): np.ndarray or list of ARIMA component
            - lstm_component (optional): np.ndarray or list of LSTM component
            - component_stats (optional): dict with component contribution percentages
                Keys: 'arima_contribution', 'lstm_contribution'
            - ticker (optional): str, asset ticker symbol
            - horizon (optional): int, forecast horizon in periods
            - timestamp (optional): str, generation timestamp

    Returns:
        str: Multi-line formatted summary string suitable for display.
            Format includes sections for:
            - Header: Timestamp and ticker information
            - Forecast Statistics: Min, max, mean, std of predictions
            - Performance Metrics: RMSE, MAE if available
            - Component Analysis: ARIMA and LSTM contributions if available
            - Model Configuration: ARIMA params, LSTM config if available

    Raises:
        TypeError: If results is not a dictionary.
        ValueError: If results is missing predictions key.

    Examples:
        >>> import numpy as np
        >>> results = {
        ...     'predictions': np.array([100.5, 101.2, 102.1, 102.8, 103.5]),
        ...     'metrics': {'rmse': 0.35, 'mae': 0.27},
        ...     'model_params': {'arima_params': (1, 1, 1), 'lstm_nodes': 10},
        ...     'arima_component': np.array([100.0, 101.0, 102.0, 102.8, 103.5]),
        ...     'lstm_component': np.array([0.5, 0.2, 0.1, 0.0, 0.0]),
        ...     'component_stats': {'arima_contribution': 95.5, 'lstm_contribution': 4.5},
        ...     'ticker': 'BTC',
        ...     'horizon': 5
        ... }
        >>> summary = format_results_summary(results)
        >>> print(summary)
    """
    try:
        # Validate inputs
        if not isinstance(results, dict):
            error_msg = f"results must be a dictionary, got {type(results)}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        if 'predictions' not in results:
            error_msg = "results missing required key 'predictions'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Formatting results summary")

        # Extract data from results
        predictions = results.get('predictions', [])
        metrics = results.get('metrics', {})
        model_params = results.get('model_params', {})
        component_stats = results.get('component_stats', {})
        ticker = results.get('ticker', 'N/A')
        horizon = results.get('horizon', len(predictions))
        timestamp = results.get('timestamp', datetime.now().isoformat())

        # Convert numpy arrays to lists if needed
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()

        # Calculate prediction statistics
        if predictions:
            pred_min = min(predictions)
            pred_max = max(predictions)
            pred_mean = sum(predictions) / len(predictions)
            pred_variance = sum((x - pred_mean) ** 2 for x in predictions) / len(predictions)
            pred_std = pred_variance ** 0.5
        else:
            pred_min = pred_max = pred_mean = pred_std = 0.0

        # Build summary string
        lines = []
        lines.append("=" * 70)
        lines.append("HYBRID LSTM-ARIMA FORECAST RESULTS SUMMARY")
        lines.append("=" * 70)

        # Header section
        lines.append(f"\nGenerated: {timestamp}")
        if ticker != 'N/A':
            lines.append(f"Ticker: {ticker}")
        lines.append(f"Forecast Horizon: {horizon} periods")

        # Forecast statistics section
        lines.append("\n" + "-" * 70)
        lines.append("FORECAST STATISTICS")
        lines.append("-" * 70)
        lines.append(f"Number of Predictions: {len(predictions)}")
        lines.append(f"Prediction Range: [{pred_min:.6f}, {pred_max:.6f}]")
        lines.append(f"Mean Value: {pred_mean:.6f}")
        lines.append(f"Standard Deviation: {pred_std:.6f}")

        # Performance metrics section
        if metrics:
            lines.append("\n" + "-" * 70)
            lines.append("PERFORMANCE METRICS")
            lines.append("-" * 70)
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    lines.append(f"{metric_name.upper():20s}: {metric_value:.6f}")
                else:
                    lines.append(f"{metric_name.upper():20s}: {metric_value}")

        # Component analysis section
        if component_stats:
            lines.append("\n" + "-" * 70)
            lines.append("COMPONENT ANALYSIS")
            lines.append("-" * 70)
            arima_contrib = component_stats.get('arima_contribution', 0)
            lstm_contrib = component_stats.get('lstm_contribution', 0)
            lines.append(f"ARIMA Contribution: {arima_contrib:.2f}%")
            lines.append(f"LSTM Contribution: {lstm_contrib:.2f}%")

            # Add component statistics if available
            if 'arima_mean' in component_stats:
                lines.append(f"\nARIMA Component:")
                lines.append(f"  Mean: {component_stats.get('arima_mean', 0):.6f}")
                lines.append(f"  Std Dev: {component_stats.get('arima_std', 0):.6f}")
            if 'lstm_mean' in component_stats:
                lines.append(f"\nLSTM Component:")
                lines.append(f"  Mean: {component_stats.get('lstm_mean', 0):.6f}")
                lines.append(f"  Std Dev: {component_stats.get('lstm_std', 0):.6f}")

        # Model configuration section
        if model_params:
            lines.append("\n" + "-" * 70)
            lines.append("MODEL CONFIGURATION")
            lines.append("-" * 70)
            for param_name, param_value in model_params.items():
                lines.append(f"{param_name:25s}: {param_value}")

        lines.append("\n" + "=" * 70)

        summary_str = "\n".join(lines)
        logger.info("Results summary formatted successfully")

        return summary_str

    except (TypeError, ValueError) as e:
        logger.error(f"Error in format_results_summary: {str(e)}")
        raise

def report_progress(current_step: int, total_steps: int, message: Optional[str] = None) -> None:
    """
    Reports progress to the console with a progress indicator.

    Args:
        current_step (int): Current step number.
        total_steps (int): Total number of steps.
        message (str, optional): Additional message to display. Defaults to None.
    """
    if current_step <= 0 or total_steps <= 0:
        logger.warning(f"Invalid progress values: current_step={current_step}, total_steps={total_steps}")
        return
    
    percentage = min(100, int((current_step / total_steps) * 100))
    bar_length = 40
    filled = int(bar_length * current_step / total_steps)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    progress_str = f"\rProgress: [{bar}] {percentage}% ({current_step}/{total_steps})"
    if message:
        progress_str += f" - {message}"
    
    print(progress_str, end='', flush=True)
    
    if current_step == total_steps:
        print()  # New line at end