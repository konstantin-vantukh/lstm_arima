"""
Data Preprocessing Module for Hybrid LSTM-ARIMA Forecasting System

This module provides functionality to load time series data, handle missing values,
calculate returns, and reshape data for LSTM processing.

Functions:
    - load_data: Load CSV/JSON cryptocurrency price data
    - impute_missing: Forward fill missing values (t with t-1)
    - calculate_returns: Calculate simple returns R_t = (P_t - P_t-1) / P_t-1
    - reshape_for_lstm: Reshape to 3D tensor [Samples, Time Steps, Features]
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load cryptocurrency price data from CSV or JSON file.

    Supports both CSV and JSON file formats. The function auto-detects the format
    based on file extension.

    Args:
        file_path (str): Path to the input file (CSV or JSON)

    Returns:
        pd.DataFrame: Loaded data as a DataFrame with price information

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is not supported (not CSV or JSON)
        Exception: For JSON parsing or CSV reading errors

    Examples:
        >>> data = load_data('data/btc_prices.csv')
        >>> data = load_data('data/eth_prices.json')
    """
    file_path = Path(file_path)

    # Validate file exists
    if not file_path.exists():
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Get file extension
    file_ext = file_path.suffix.lower()

    try:
        if file_ext == ".csv":
            logger.info(f"Loading CSV data from {file_path}")
            data = pd.read_csv(file_path)
        elif file_ext == ".json":
            logger.info(f"Loading JSON data from {file_path}")
            with open(file_path, "r") as f:
                json_data = json.load(f)
            data = pd.DataFrame(json_data)
        else:
            error_msg = f"Unsupported file format: {file_ext}. Only CSV and JSON are supported."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Data shape: {data.shape}")
        return data

    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing error in {file_path}: {str(e)}"
        logger.error(error_msg)
        raise
    except pd.errors.ParserError as e:
        error_msg = f"CSV parsing error in {file_path}: {str(e)}"
        logger.error(error_msg)
        raise


def impute_missing(data: pd.Series) -> pd.Series:
    """
    Forward fill missing values in a time series.

    Replaces missing values (NaN) with the previous available value (t-1).
    If no previous value exists, fills backward. This ensures data continuity
    for time series analysis.

    Args:
        data (pd.Series): Input time series with potential missing values

    Returns:
        pd.Series: Series with missing values filled using forward fill method

    Examples:
        >>> prices = pd.Series([100, np.nan, np.nan, 102, 103])
        >>> imputed = impute_missing(prices)
        >>> imputed.tolist()
        [100.0, 100.0, 100.0, 102.0, 103.0]
    """
    logger.info(f"Imputing missing values. Missing count before: {data.isna().sum()}")

    # Create a copy to avoid modifying the original
    imputed_data = data.copy()

    # Forward fill missing values
    imputed_data = imputed_data.fillna(method="ffill")

    # If there are still missing values (at the beginning), backward fill
    imputed_data = imputed_data.fillna(method="bfill")

    missing_after = imputed_data.isna().sum()
    logger.info(f"Missing values after imputation: {missing_after}")

    if missing_after > 0:
        logger.warning(
            f"Still {missing_after} missing values after forward and backward fill"
        )

    return imputed_data


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns from a price series.

    Computes percentage returns using the formula: R_t = (P_t - P_{t-1}) / P_{t-1}
    This converts raw price data to a scale-free dataset suitable for modeling.

    Args:
        prices (pd.Series): Input price series

    Returns:
        pd.Series: Calculated returns series (first value is NaN)

    Raises:
        ValueError: If prices contain zero or negative values

    Examples:
        >>> prices = pd.Series([100, 102, 101, 103])
        >>> returns = calculate_returns(prices)
        >>> returns.tolist()
        [nan, 0.02, -0.0098..., 0.0198...]
    """
    logger.info(f"Calculating returns from price series of length {len(prices)}")

    # Validate prices are positive
    if (prices <= 0).any():
        error_msg = "Price series contains zero or negative values"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Calculate simple returns: (P_t - P_{t-1}) / P_{t-1}
    returns = prices.pct_change()

    logger.info(f"Returns calculated. Mean: {returns.mean():.6f}, Std: {returns.std():.6f}")

    return returns


def reshape_for_lstm(
    data: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape data into 3D tensor format for LSTM input.

    Converts a 1D array into 3D tensor of shape [Samples, Time Steps, Features]
    using a rolling window approach. Each sample contains a window of consecutive
    time steps with a single feature (univariate).

    Args:
        data (np.ndarray): 1D array of time series data
        window_size (int): Size of the rolling window (time steps)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of (X, y) where:
            - X: 3D array of shape [Samples, window_size, 1] - input sequences
            - y: 1D array of shape [Samples] - target values (next time step)

    Raises:
        ValueError: If window_size is invalid or data is too short

    Examples:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> X, y = reshape_for_lstm(data, window_size=3)
        >>> X.shape
        (7, 3, 1)
        >>> y.shape
        (7,)
    """
    if not isinstance(data, np.ndarray):
        error_msg = "Data must be a numpy array"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(data.shape) != 1:
        error_msg = "Data must be 1D array"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if window_size <= 0:
        error_msg = f"Window size must be positive, got {window_size}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(data) < window_size + 1:
        error_msg = (
            f"Data length ({len(data)}) must be greater than window_size + 1 ({window_size + 1})"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(
        f"Reshaping data for LSTM. Data length: {len(data)}, Window size: {window_size}"
    )

    X = []
    y = []

    # Create rolling windows
    for i in range(len(data) - window_size):
        # Extract window and reshape to [window_size, 1]
        window = data[i : i + window_size].reshape(-1, 1)
        X.append(window)

        # Next value is the target
        y.append(data[i + window_size])

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Reshaped data. X shape: {X.shape}, y shape: {y.shape}")

    return X, y
