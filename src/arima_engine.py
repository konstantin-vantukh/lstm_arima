"""
ARIMA Engine Module for Hybrid LSTM-ARIMA Forecasting System

This module provides ARIMA (AutoRegressive Integrated Moving Average) functionality
for time series forecasting. It handles stationarity testing, parameter optimization,
model fitting, and residual extraction.

Functions:
    - test_stationarity: Perform Augmented Dickey-Fuller (ADF) test
    - find_optimal_params: Auto-ARIMA parameter selection with AIC minimization
    - fit_arima: Fit ARIMA model with specified (p, d, q) parameters
    - extract_residuals: Calculate residuals as actual - predicted values
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import aic, bic


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_stationarity(series: pd.Series) -> Tuple[bool, float]:
    """
    Perform Augmented Dickey-Fuller (ADF) test on a time series.

    Tests whether the series has a unit root (non-stationary) using the ADF test.
    The null hypothesis (H0) is that the series has a unit root (is non-stationary).
    If p-value < 0.05, we reject H0 and conclude the series is stationary.

    Args:
        series (pd.Series): Input time series to test for stationarity

    Returns:
        Tuple[bool, float]: A tuple containing:
            - is_stationary (bool): True if series is stationary (p-value < 0.05), False otherwise
            - p_value (float): The p-value from the ADF test

    Raises:
        ValueError: If the input series is empty or contains NaN values
        Exception: For any statsmodels ADF test failures

    Examples:
        >>> prices = pd.Series([100, 101, 102, 103, 104])
        >>> is_stat, p_val = test_stationarity(prices)
        >>> print(f"Stationary: {is_stat}, p-value: {p_val:.4f}")
    """
    # Validate input
    if len(series) == 0:
        error_msg = "Series is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if series.isna().any():
        error_msg = "Series contains NaN values"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.info(f"Performing ADF test on series of length {len(series)}")

        # Perform ADF test
        adf_result = adfuller(series, autolag="AIC")

        # Extract p-value
        p_value = adf_result[1]

        # Determine stationarity (p-value < 0.05 indicates stationarity)
        is_stationary = p_value < 0.05

        logger.info(f"ADF Test Results - Test Statistic: {adf_result[0]:.6f}, p-value: {p_value:.6f}, Stationary: {is_stationary}")

        return is_stationary, p_value

    except Exception as e:
        error_msg = f"ADF test failed: {str(e)}"
        logger.error(error_msg)
        raise


def find_optimal_params(
    series: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5
) -> Tuple[int, int, int]:
    """
    Find optimal ARIMA parameters (p, d, q) using auto-ARIMA with AIC minimization.

    Searches through the parameter space defined by max_p, max_d, and max_q to find
    the (p, d, q) combination that minimizes the Akaike Information Criterion (AIC).
    The function applies differencing iteratively until the series becomes stationary
    (up to max_d times).

    Args:
        series (pd.Series): Input time series to analyze
        max_p (int): Maximum AR order to test (default: 5)
        max_d (int): Maximum differencing order (default: 2, limited for stability)
        max_q (int): Maximum MA order to test (default: 5)

    Returns:
        Tuple[int, int, int]: Optimal (p, d, q) parameters

    Raises:
        ValueError: If series is empty or parameters are invalid
        Exception: If all parameter combinations fail to converge

    Note:
        - d is limited to max 2 for model stability as per architecture requirements
        - AIC is used as the information criterion for model selection
        - If convergence fails for all combinations, returns default (1, 1, 1)

    Examples:
        >>> returns = pd.Series(np.random.randn(100))
        >>> p, d, q = find_optimal_params(returns, max_p=3, max_d=2, max_q=3)
        >>> print(f"Optimal ARIMA order: ({p}, {d}, {q})")
    """
    # Validate input
    if len(series) == 0:
        error_msg = "Series is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if max_p < 0 or max_d < 0 or max_q < 0:
        error_msg = "max_p, max_d, max_q must be non-negative"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if max_d > 2:
        logger.warning(f"max_d={max_d} exceeds recommended limit of 2. Setting to 2 for stability.")
        max_d = 2

    try:
        logger.info(
            f"Starting auto-ARIMA search. Max P: {max_p}, Max D: {max_d}, Max Q: {max_q}, Series length: {len(series)}"
        )

        best_aic = float("inf")
        best_order = (1, 1, 1)  # Default order

        # Grid search over parameter space
        for d in range(max_d + 1):
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    try:
                        # Fit ARIMA model with current parameters
                        model = ARIMA(series, order=(p, d, q))
                        results = model.fit()

                        # Extract AIC value
                        current_aic = results.aic

                        logger.debug(f"Order ({p}, {d}, {q}): AIC = {current_aic:.4f}")

                        # Update best parameters if this AIC is lower
                        if current_aic < best_aic:
                            best_aic = current_aic
                            best_order = (p, d, q)

                    except Exception as e:
                        # Log convergence failures but continue searching
                        logger.debug(f"Order ({p}, {d}, {q}) failed to converge: {str(e)}")
                        continue

        logger.info(f"Optimal ARIMA order found: {best_order} with AIC: {best_aic:.4f}")

        return best_order

    except Exception as e:
        error_msg = f"Auto-ARIMA search failed: {str(e)}"
        logger.error(error_msg)
        logger.warning("Returning default ARIMA order (1, 1, 1)")
        return (1, 1, 1)


def fit_arima(series: pd.Series, order: Tuple[int, int, int]):
    """
    Fit an ARIMA model to a time series with specified (p, d, q) parameters.

    Creates and fits an ARIMA model using statsmodels' ARIMA implementation.
    This function handles the actual model fitting process and returns the
    fitted model results object.

    Args:
        series (pd.Series): Input time series to model
        order (Tuple[int, int, int]): ARIMA parameters (p, d, q)

    Returns:
        statsmodels.tsa.arima.model.ARIMAResults: Fitted ARIMA model results object
            containing model parameters, diagnostics, and forecasting methods

    Raises:
        ValueError: If series is empty or order parameters are invalid
        Exception: If ARIMA model fails to converge

    Examples:
        >>> returns = pd.Series(np.random.randn(100))
        >>> model = fit_arima(returns, order=(1, 1, 1))
        >>> forecast = model.get_forecast(steps=10)
    """
    # Validate inputs
    if len(series) == 0:
        error_msg = "Series is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    p, d, q = order
    if p < 0 or d < 0 or q < 0:
        error_msg = f"Invalid ARIMA order: {order}. All parameters must be non-negative"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if d > 2:
        logger.warning(f"Differencing order d={d} exceeds recommended limit of 2")

    try:
        logger.info(f"Fitting ARIMA model with order {order} on series of length {len(series)}")

        # Create and fit ARIMA model
        model = ARIMA(series, order=order)
        results = model.fit()

        logger.info(f"ARIMA model fitted successfully. AIC: {results.aic:.4f}, BIC: {results.bic:.4f}")

        return results

    except Exception as e:
        error_msg = f"ARIMA model fitting failed with order {order}: {str(e)}"
        logger.error(error_msg)
        raise


def extract_residuals(series: pd.Series, arima_model) -> pd.Series:
    """
    Extract residuals from fitted ARIMA model as actual - predicted values.

    Calculates the in-sample residuals by subtracting the model's fitted values
    from the actual series values. Residuals represent the unexplained variation
    after removing the linear ARIMA component, which will be modeled by the LSTM.

    Args:
        series (pd.Series): Original input time series
        arima_model: Fitted ARIMA model results object (from fit_arima)

    Returns:
        pd.Series: Residuals as (actual - predicted) values with preserved index

    Raises:
        ValueError: If series is empty or lengths don't match
        AttributeError: If arima_model lacks required attributes

    Examples:
        >>> returns = pd.Series(np.random.randn(100))
        >>> model = fit_arima(returns, order=(1, 1, 1))
        >>> residuals = extract_residuals(returns, model)
        >>> print(f"Residuals shape: {residuals.shape}, Mean: {residuals.mean():.6f}")

    Note:
        - Residuals have length equal to the original series
        - Used as input for LSTM to model non-linear patterns
        - For differenced series, residuals are calculated on original scale
    """
    # Validate inputs
    if len(series) == 0:
        error_msg = "Series is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not hasattr(arima_model, "fittedvalues"):
        error_msg = "Invalid ARIMA model object - missing fittedvalues attribute"
        logger.error(error_msg)
        raise AttributeError(error_msg)

    try:
        logger.info(f"Extracting residuals from ARIMA model")

        # Get fitted values from the model
        fitted_values = arima_model.fittedvalues

        # Verify lengths match
        if len(series) != len(fitted_values):
            error_msg = (
                f"Series length ({len(series)}) does not match fitted values length ({len(fitted_values)})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate residuals as actual - predicted
        residuals = series - fitted_values

        logger.info(
            f"Residuals extracted. Length: {len(residuals)}, Mean: {residuals.mean():.6f}, "
            f"Std: {residuals.std():.6f}, Min: {residuals.min():.6f}, Max: {residuals.max():.6f}"
        )

        return residuals

    except Exception as e:
        error_msg = f"Failed to extract residuals: {str(e)}"
        logger.error(error_msg)
        raise
