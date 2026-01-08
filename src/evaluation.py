"""
Evaluation Module for Hybrid LSTM-ARIMA Forecasting System

This module provides functionality to calculate performance metrics (RMSE, MAE),
perform walk-forward validation for time series models, and generate comprehensive
metrics reports.

Functions:
    - calculate_rmse: Calculate Root Mean Squared Error
    - calculate_mae: Calculate Mean Absolute Error
    - walk_forward_validation: Perform walk-forward validation on time series data
    - create_metrics_report: Generate comprehensive metrics report with metadata
"""

import logging
import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple, Union, Any

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) between actual and predicted values.

    RMSE is calculated as: √(mean of squared errors)
    This metric penalizes large errors more heavily than small ones.

    Args:
        actual (np.ndarray): Array of actual/observed values
        predicted (np.ndarray): Array of predicted values

    Returns:
        float: Root Mean Squared Error value

    Raises:
        ValueError: If arrays are empty, have mismatched lengths, or contain invalid values
        TypeError: If input is not a numpy array or cannot be converted

    Examples:
        >>> actual = np.array([1, 2, 3, 4, 5])
        >>> predicted = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        >>> rmse = calculate_rmse(actual, predicted)
        >>> rmse
        0.12649...
    """
    try:
        # Convert to numpy arrays if needed
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)

        # Validate inputs
        if actual.size == 0 or predicted.size == 0:
            error_msg = "Input arrays cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if actual.shape != predicted.shape:
            error_msg = (
                f"Mismatched array lengths: actual ({actual.shape}) vs predicted ({predicted.shape})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for NaN values
        if np.any(np.isnan(actual)) or np.any(np.isnan(predicted)):
            error_msg = "Input arrays contain NaN values"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate squared errors
        squared_errors = (actual - predicted) ** 2

        # Calculate mean of squared errors
        mse = np.mean(squared_errors)

        # Calculate RMSE
        rmse = np.sqrt(mse)

        logger.info(f"RMSE calculated: {rmse:.6f}")
        return float(rmse)

    except TypeError as e:
        error_msg = f"Type error in calculate_rmse: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE) between actual and predicted values.

    MAE is calculated as: mean of absolute errors
    This metric treats all errors equally regardless of magnitude.

    Args:
        actual (np.ndarray): Array of actual/observed values
        predicted (np.ndarray): Array of predicted values

    Returns:
        float: Mean Absolute Error value

    Raises:
        ValueError: If arrays are empty, have mismatched lengths, or contain invalid values
        TypeError: If input is not a numpy array or cannot be converted

    Examples:
        >>> actual = np.array([1, 2, 3, 4, 5])
        >>> predicted = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        >>> mae = calculate_mae(actual, predicted)
        >>> mae
        0.12
    """
    try:
        # Convert to numpy arrays if needed
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)

        # Validate inputs
        if actual.size == 0 or predicted.size == 0:
            error_msg = "Input arrays cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if actual.shape != predicted.shape:
            error_msg = (
                f"Mismatched array lengths: actual ({actual.shape}) vs predicted ({predicted.shape})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for NaN values
        if np.any(np.isnan(actual)) or np.any(np.isnan(predicted)):
            error_msg = "Input arrays contain NaN values"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate absolute errors
        absolute_errors = np.abs(actual - predicted)

        # Calculate mean of absolute errors
        mae = np.mean(absolute_errors)

        logger.info(f"MAE calculated: {mae:.6f}")
        return float(mae)

    except TypeError as e:
        error_msg = f"Type error in calculate_mae: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def walk_forward_validation(
    data: pd.Series, model_func: Callable, test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Perform walk-forward validation on time series data.

    Walk-forward validation trains on historic data and tests sequentially on
    out-of-sample data, moving the training window forward one step at a time.
    This prevents temporal data leakage and provides realistic performance estimates.

    Process:
    1. Calculate initial train/test split point based on test_size
    2. For each step in the test period:
       a. Train model on historical data (from start to current point)
       b. Predict next point
       c. Move window forward

    Args:
        data (pd.Series): Time series data to validate (must be sorted chronologically)
        model_func (Callable): Function signature: model_func(train_data) -> predictions
                               Should return model object or predictor function
        test_size (float): Fraction of data to use for testing (default: 0.2, range: 0.1-0.5)

    Returns:
        dict: Validation results containing:
            - predictions: np.ndarray of all predictions from walk-forward passes
            - actuals: np.ndarray of actual values corresponding to predictions
            - rmse: float, Root Mean Squared Error across all predictions
            - mae: float, Mean Absolute Error across all predictions
            - test_size: float, Proportion of data used for testing
            - num_iterations: int, Number of validation iterations performed
            - metadata: dict containing timestamps and split information

    Raises:
        ValueError: If test_size is invalid, data is too short, or model_func fails
        TypeError: If inputs have incorrect types

    Examples:
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression
        >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> def simple_model(train_data):
        ...     # Train model and return predictions for next point
        ...     return train_data.iloc[-1] * 1.05
        >>> results = walk_forward_validation(data, simple_model, test_size=0.3)
    """
    try:
        # Validate inputs
        if not isinstance(data, pd.Series):
            error_msg = "Data must be a pandas Series"
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not callable(model_func):
            error_msg = "model_func must be callable"
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not (0.1 <= test_size <= 0.5):
            error_msg = f"test_size must be between 0.1 and 0.5, got {test_size}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for NaN values
        if data.isna().any():
            error_msg = "Data contains NaN values"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate split point
        n_samples = len(data)
        n_test = max(1, int(n_samples * test_size))
        n_train = n_samples - n_test

        if n_train < 2:
            error_msg = (
                f"Not enough training data: need at least 2 samples, got {n_train} "
                f"(test_size={test_size} with {n_samples} total samples)"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"Starting walk-forward validation: total={n_samples}, train={n_train}, test={n_test}"
        )

        # Storage for predictions and actuals
        predictions = []
        actuals = []
        errors_log = []

        # Walk-forward validation loop
        for i in range(n_test):
            # Define train and test split
            train_idx = n_train + i
            test_idx = train_idx + 1

            # Extract training data (from start to current point)
            train_data = data.iloc[:train_idx]

            # Actual value to predict
            actual_value = data.iloc[train_idx]

            try:
                # Call model function with training data
                # Expected return: single prediction value (float)
                prediction = model_func(train_data)

                # Handle case where model returns array or Series
                if isinstance(prediction, (np.ndarray, pd.Series)):
                    if len(prediction) > 0:
                        prediction = float(prediction[-1] if len(prediction) > 0 else prediction)
                    else:
                        raise ValueError("Model returned empty array")
                else:
                    prediction = float(prediction)

                predictions.append(prediction)
                actuals.append(actual_value)
                errors_log.append(
                    {
                        "iteration": i + 1,
                        "train_size": train_idx,
                        "actual": actual_value,
                        "predicted": prediction,
                        "error": actual_value - prediction,
                    }
                )

                logger.debug(
                    f"Iteration {i+1}: train_size={train_idx}, actual={actual_value:.6f}, "
                    f"predicted={prediction:.6f}"
                )

            except Exception as e:
                error_msg = f"Model function failed at iteration {i+1}: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e

        # Convert to numpy arrays
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)

        # Calculate metrics
        rmse = calculate_rmse(actuals_array, predictions_array)
        mae = calculate_mae(actuals_array, predictions_array)

        logger.info(f"Walk-forward validation complete. RMSE: {rmse:.6f}, MAE: {mae:.6f}")

        # Prepare results dictionary
        results = {
            "predictions": predictions_array,
            "actuals": actuals_array,
            "rmse": rmse,
            "mae": mae,
            "test_size": test_size,
            "num_iterations": n_test,
            "metadata": {
                "total_samples": n_samples,
                "train_samples": n_train,
                "test_samples": n_test,
                "errors_log": errors_log,
            },
        }

        return results

    except (TypeError, ValueError) as e:
        logger.error(f"Error in walk_forward_validation: {str(e)}")
        raise


def create_metrics_report(
    actual: np.ndarray, predicted: np.ndarray, model_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive metrics report for model predictions.

    This function calculates performance metrics (RMSE, MAE) and compiles them
    into a structured report with metadata about the model and predictions.

    Args:
        actual (np.ndarray): Array of actual/observed values
        predicted (np.ndarray): Array of predicted values
        model_params (dict, optional): Dictionary of model parameters to include in report
                                       Example: {"arima_params": (1, 1, 1), "lstm_nodes": 10}

    Returns:
        dict: Comprehensive metrics report containing:
            - metrics: dict with calculated metrics
                - rmse: float, Root Mean Squared Error
                - mae: float, Mean Absolute Error
            - predictions: dict with prediction statistics
                - actual_mean: float, Mean of actual values
                - actual_std: float, Standard deviation of actual values
                - predicted_mean: float, Mean of predicted values
                - predicted_std: float, Standard deviation of predicted values
                - n_predictions: int, Number of predictions
            - model_params: dict, Model parameters provided by user
            - summary: str, Human-readable summary of metrics

    Raises:
        ValueError: If arrays are invalid or calculations fail

    Examples:
        >>> actual = np.array([1, 2, 3, 4, 5])
        >>> predicted = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        >>> params = {"model_type": "hybrid", "arima_params": (1, 1, 1)}
        >>> report = create_metrics_report(actual, predicted, model_params=params)
        >>> report["metrics"]["rmse"]
        0.12649...
    """
    try:
        # Convert to numpy arrays
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)

        # Validate inputs
        if actual.size == 0 or predicted.size == 0:
            error_msg = "Input arrays cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if actual.shape != predicted.shape:
            error_msg = (
                f"Mismatched array lengths: actual ({actual.shape}) vs predicted ({predicted.shape})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for NaN values
        if np.any(np.isnan(actual)) or np.any(np.isnan(predicted)):
            error_msg = "Input arrays contain NaN values"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"Creating metrics report for {len(actual)} predictions"
        )

        # Calculate metrics
        rmse = calculate_rmse(actual, predicted)
        mae = calculate_mae(actual, predicted)

        # Calculate prediction statistics
        actual_mean = float(np.mean(actual))
        actual_std = float(np.std(actual))
        predicted_mean = float(np.mean(predicted))
        predicted_std = float(np.std(predicted))
        n_predictions = len(actual)

        # Handle edge case where model_params is None
        if model_params is None:
            model_params = {}
        else:
            model_params = dict(model_params)  # Create a copy

        # Create summary string
        summary = (
            f"Model Performance Summary: "
            f"RMSE={rmse:.6f}, MAE={mae:.6f}, "
            f"Predictions={n_predictions}, "
            f"Actual Mean={actual_mean:.6f}, "
            f"Predicted Mean={predicted_mean:.6f}"
        )

        # Compile comprehensive report
        report = {
            "metrics": {
                "rmse": rmse,
                "mae": mae,
            },
            "predictions": {
                "actual_mean": actual_mean,
                "actual_std": actual_std,
                "predicted_mean": predicted_mean,
                "predicted_std": predicted_std,
                "n_predictions": n_predictions,
            },
            "model_params": model_params,
            "summary": summary,
        }

        logger.info(f"Metrics report created: {summary}")

        return report

    except ValueError as e:
        logger.error(f"Error in create_metrics_report: {str(e)}")
        raise
