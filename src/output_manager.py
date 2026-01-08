"""
Output Manager Module for Hybrid LSTM-ARIMA Forecasting System

This module handles exporting results and reporting progress during the forecasting
pipeline. It provides functionality to export predictions and metrics to CSV/JSON formats,
log training progress to STDOUT, and generate human-readable forecast summaries.

Functions:
    - export_results: Export results to CSV or JSON file
    - log_progress: Log training progress to STDOUT
    - format_results_summary: Create human-readable summary of forecast results
"""

import logging
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def log_progress(
    epoch: int,
    loss: float,
    stage: str = 'training'
) -> None:
    """
    Log training progress to STDOUT.

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
        >>> log_progress(1, 0.5234567, stage='training')
        >>> # Output to STDOUT: [Epoch 1 | training] Loss: 0.523457

        >>> log_progress(10, 0.0987654, stage='validation')
        >>> # Output to STDOUT: [Epoch 10 | validation] Loss: 0.098765

        >>> for epoch in range(1, 6):
        ...     loss = 0.5 / (epoch + 1)
        ...     log_progress(epoch, loss)
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
        logger.error(f"Error in log_progress: {str(e)}")
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
