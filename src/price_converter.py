"""
Price Converter Module for Hybrid LSTM-ARIMA Forecasting System

This module handles the conversion of returns-space forecasts to price-space
reconstructions. It implements the inverse simple returns formula to convert
predicted percentage changes back to absolute prices.

Key Formulas:
    Single-step: P̂_t = P_{t-1} × (1 + R̂_t)
    Multi-step:  P̂_{t+i} = P̂_{t+i-1} × (1 + R̂_{t+i})

Where:
    P̂_t = Reconstructed price at time t
    P_{t-1} = Last observed price (previous time step)
    R̂_t = Predicted returns (percentage change) at time t

Functions:
    - reconstruct_price_single: Single-step price reconstruction
    - reconstruct_price_series: Multi-step price reconstruction with compounding
    - validate_reconstructed_prices: Validation and diagnostics

Error Handling & Logging: Section 10, Error Handling Strategy
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple

from src.exceptions import PriceConversionError
from src.logger_config import get_logger, log_exception

# Configure module logging
logger = get_logger(__name__)


def reconstruct_price_single(
    last_price: float,
    returns_value: float
) -> float:
    """
    Reconstruct a single price from a returns-space prediction.

    Converts a predicted return (percentage change) back to absolute price
    using the inverse simple returns formula: P̂_t = P_{t-1} × (1 + R̂_t)

    Args:
        last_price (float): Previous price (P_{t-1}) - the base price
            Must be positive and finite
            Represents the most recent observed price
        returns_value (float): Predicted return (R̂_t) - percentage change
            Represents the predicted percentage change from last_price
            No specific bounds, but typically in range [-1, inf)
            Example: 0.02 means +2%, -0.05 means -5%

    Returns:
        float: Reconstructed price (P̂_t) at current time step
            Calculated as: last_price × (1 + returns_value)
            Example: last_price=100, returns_value=0.02 → returns 102.0

    Raises:
        ValueError: If last_price <= 0 or contains NaN/inf
        TypeError: If inputs are not numeric types

    Examples:
        >>> last_price = 100.0
        >>> returns_value = 0.02
        >>> reconstructed = reconstruct_price_single(last_price, returns_value)
        >>> print(f"${reconstructed:.2f}")
        $102.00

        >>> last_price = 50000.0
        >>> returns_value = -0.015
        >>> reconstructed = reconstruct_price_single(last_price, returns_value)
        >>> print(f"${reconstructed:.2f}")
        $49250.00

    Note:
        - This function performs single-step reconstruction only
        - For multi-step forecasts, use reconstruct_price_series()
        - Negative prices may result if returns_value <= -1
          (e.g., -100% return = zero or negative price)
        - Returns are compounded multiplicatively, not additively
    """
    # Input validation - type checks
    if not isinstance(last_price, (int, float, np.number)):
        error_msg = f"last_price must be numeric, got {type(last_price)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if not isinstance(returns_value, (int, float, np.number)):
        error_msg = f"returns_value must be numeric, got {type(returns_value)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Input validation - NaN/inf checks
    if np.isnan(last_price) or np.isinf(last_price):
        error_msg = f"last_price contains NaN or inf: {last_price}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if np.isnan(returns_value) or np.isinf(returns_value):
        error_msg = f"returns_value contains NaN or inf: {returns_value}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Input validation - last_price must be positive
    if last_price <= 0:
        error_msg = f"last_price must be positive, got {last_price}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Convert to float for computation
        last_price_f = float(last_price)
        returns_value_f = float(returns_value)

        # Apply price reconstruction formula: P̂_t = P_{t-1} × (1 + R̂_t)
        reconstructed_price = last_price_f * (1 + returns_value_f)

        logger.debug(
            f"Single-step price reconstruction: "
            f"last_price={last_price_f:.6f}, "
            f"returns={returns_value_f:.6f}, "
            f"reconstructed={reconstructed_price:.6f}"
        )

        return float(reconstructed_price)

    except (ValueError, TypeError) as e:
        error_msg = f"Input validation failed in single price reconstruction: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise PriceConversionError(
            error_message=error_msg,
            prices=np.array([last_price]),
            formula="P = last_price * (1 + returns)"
        ) from e
    except Exception as e:
        error_msg = f"Failed to reconstruct single price: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise PriceConversionError(
            error_message=error_msg,
            prices=np.array([last_price]),
            formula="P = last_price * (1 + returns)"
        ) from e


def reconstruct_price_series(
    last_price: float,
    returns_forecast: np.ndarray
) -> np.ndarray:
    """
    Reconstruct a series of prices from multi-step returns predictions.

    Iteratively reconstructs prices from a returns-space forecast using
    the compounding formula. Each step builds upon the previous reconstructed
    price, creating a continuous price path from the starting point.

    Multi-step reconstruction compounding:
        P̂_{t+1} = P_t × (1 + R̂_{t+1})
        P̂_{t+2} = P̂_{t+1} × (1 + R̂_{t+2})
        ...
        P̂_{t+h} = P̂_{t+h-1} × (1 + R̂_{t+h})

    Args:
        last_price (float): Starting price (P_t) - the base price
            Must be positive and finite
            This is the price at time t (before the forecast horizon)
            Typically the last observed historical price
        returns_forecast (np.ndarray): Array of predicted returns
            Shape: (horizon,) - 1D array of return predictions
            Each element is a predicted return (percentage change)
            Example: [0.01, -0.02, 0.015] = [+1%, -2%, +1.5%]
            All elements must be numeric, finite, no NaN/inf

    Returns:
        np.ndarray: Reconstructed price series (P̂_t for each horizon step)
            Shape: (horizon,) - same length as returns_forecast
            Each element is the reconstructed price at that time step
            Prices compound iteratively from the starting last_price

    Raises:
        ValueError: If last_price <= 0, returns_forecast is empty,
            contains NaN/inf, or has incompatible dimensions
        TypeError: If inputs are not correct types

    Examples:
        >>> last_price = 100.0
        >>> returns = np.array([0.02, -0.01, 0.03])
        >>> prices = reconstruct_price_series(last_price, returns)
        >>> print(prices)
        [102.0  101.0  104.06]

        >>> last_price = 50000.0
        >>> returns = np.array([0.015, -0.005, 0.01])
        >>> prices = reconstruct_price_series(last_price, returns)
        # P̂_1 = 50000 * (1 + 0.015) = 50750
        # P̂_2 = 50750 * (1 - 0.005) = 50499.25
        # P̂_3 = 50499.25 * (1 + 0.01) = 51004.25

    Note:
        - Each step compounds on the PREVIOUS RECONSTRUCTED PRICE, not the base price
        - This creates exponential compounding of returns
        - Negative prices may result if cumulative returns drop below -100%
        - Use validate_reconstructed_prices() to check for negative values
        - Output length always matches input length
        - No NaN or inf values in input are allowed
    """
    # Input validation - type checks
    if not isinstance(last_price, (int, float, np.number)):
        error_msg = f"last_price must be numeric, got {type(last_price)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if not isinstance(returns_forecast, np.ndarray):
        error_msg = f"returns_forecast must be numpy array, got {type(returns_forecast)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Input validation - numeric type checks
    if not np.issubdtype(returns_forecast.dtype, np.number):
        error_msg = f"returns_forecast must contain numeric values, got dtype {returns_forecast.dtype}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Input validation - dimensionality checks
    if returns_forecast.ndim != 1:
        error_msg = f"returns_forecast must be 1D array, got shape {returns_forecast.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Input validation - empty checks
    if len(returns_forecast) == 0:
        error_msg = "returns_forecast is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Input validation - NaN/inf checks for last_price
    if np.isnan(last_price) or np.isinf(last_price):
        error_msg = f"last_price contains NaN or inf: {last_price}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Input validation - last_price must be positive
    if last_price <= 0:
        error_msg = f"last_price must be positive, got {last_price}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Input validation - NaN/inf checks for returns_forecast
    if np.isnan(returns_forecast).any() or np.isinf(returns_forecast).any():
        error_msg = "returns_forecast contains NaN or inf values"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.info(
            f"Reconstructing price series: "
            f"last_price={last_price:.6f}, "
            f"forecast_length={len(returns_forecast)}"
        )

        # Initialize output array
        horizon = len(returns_forecast)
        reconstructed_prices = np.zeros(horizon, dtype=np.float64)

        # Start with last_price and compound iteratively
        current_price = float(last_price)

        for i in range(horizon):
            # Apply single-step reconstruction: P̂_{t+i} = P̂_{t+i-1} × (1 + R̂_{t+i})
            current_price = current_price * (1 + float(returns_forecast[i]))
            reconstructed_prices[i] = current_price

            logger.debug(
                f"Step {i+1}/{horizon}: "
                f"return={returns_forecast[i]:.6f}, "
                f"price={current_price:.6f}"
            )

        logger.info(
            f"Price series reconstruction completed: "
            f"initial_price={last_price:.6f}, "
            f"final_price={reconstructed_prices[-1]:.6f}, "
            f"min_price={reconstructed_prices.min():.6f}, "
            f"max_price={reconstructed_prices.max():.6f}"
        )

        return reconstructed_prices

    except (ValueError, TypeError) as e:
        error_msg = f"Input validation failed in series price reconstruction: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise PriceConversionError(
            error_message=error_msg,
            prices=returns_forecast,
            formula="P_{t+i} = P_{t+i-1} * (1 + R_{t+i})"
        ) from e
    except Exception as e:
        error_msg = f"Failed to reconstruct price series: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        # Check for negative prices and log warning
        if 'reconstructed_prices' in locals() and np.any(reconstructed_prices <= 0):
            logger.warning(f"Price reconstruction resulted in negative/zero prices: min={reconstructed_prices.min():.6f}")
            raise PriceConversionError(
                error_message="Price reconstruction resulted in negative prices",
                prices=reconstructed_prices,
                formula="P_{t+i} = P_{t+i-1} * (1 + R_{t+i})"
            ) from e
        raise PriceConversionError(
            error_message=error_msg,
            prices=returns_forecast,
            formula="P_{t+i} = P_{t+i-1} * (1 + R_{t+i})"
        ) from e


def validate_reconstructed_prices(
    prices: np.ndarray
) -> Dict[str, Any]:
    """
    Validate and generate diagnostics for reconstructed prices.

    Checks reconstructed prices for validity and generates comprehensive
    statistics, warnings, and diagnostics for quality assurance.

    Args:
        prices (np.ndarray): Reconstructed price series to validate
            Shape: (n_steps,) - 1D array of price values
            Typically output from reconstruct_price_series()
            Should contain numeric values

    Returns:
        Dict[str, Any]: Comprehensive validation report containing:
            - is_valid (bool): True if all prices are valid (> 0, no NaN/inf)
            - has_negative_prices (bool): True if any prices <= 0
            - has_nan_values (bool): True if any NaN values present
            - has_inf_values (bool): True if any inf values present
            - statistics (dict): Price statistics:
                - min (float): Minimum price
                - max (float): Maximum price
                - mean (float): Mean price
                - std (float): Standard deviation
                - median (float): Median (50th percentile)
                - q25 (float): 25th percentile
                - q75 (float): 75th percentile
            - price_range (dict): Range information:
                - range (float): Max - Min
                - percent_change (float): Percentage change from min to max
                - first_price (float): First price in series
                - last_price (float): Last price in series
            - warnings (list): List of string warnings for issues found
                - "Contains negative prices" if any(prices <= 0)
                - "Contains NaN values" if any(np.isnan(prices))
                - "Contains infinity values" if any(np.isinf(prices))
                - "Prices show extreme volatility" if std > 2*mean
            - messages (list): Informational messages summarizing validation
                - Price validity status
                - Statistics summary
                - Recommendations if issues found

    Raises:
        ValueError: If prices array is empty or not 1D
        TypeError: If prices is not a numpy array or not numeric

    Examples:
        >>> prices = np.array([100.0, 102.0, 101.5, 103.5])
        >>> report = validate_reconstructed_prices(prices)
        >>> print(f"Valid: {report['is_valid']}")
        Valid: True
        >>> print(f"Mean price: ${report['statistics']['mean']:.2f}")
        Mean price: $101.75

        >>> prices_with_negative = np.array([100.0, 50.0, -10.0, 45.0])
        >>> report = validate_reconstructed_prices(prices_with_negative)
        >>> print(f"Valid: {report['is_valid']}")
        Valid: False
        >>> print(report['warnings'])
        ['Contains negative prices']

    Note:
        - Negative prices are flagged as invalid (prices should be > 0)
        - NaN and inf values are flagged as errors
        - This is a diagnostic/validation function only, doesn't raise on invalid input
        - Use this to validate before using reconstructed prices in downstream tasks
        - All statistics calculated only from valid (finite, positive) values
    """
    # Input validation - type checks
    if not isinstance(prices, np.ndarray):
        error_msg = f"prices must be numpy array, got {type(prices)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Input validation - numeric type checks
    if not np.issubdtype(prices.dtype, np.number):
        error_msg = f"prices must contain numeric values, got dtype {prices.dtype}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Input validation - dimensionality checks
    if prices.ndim != 1:
        error_msg = f"prices must be 1D array, got shape {prices.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Input validation - empty checks
    if len(prices) == 0:
        error_msg = "prices array is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.info(f"Validating reconstructed prices: length={len(prices)}")

        # Check for NaN, inf, and negative values
        has_negative = np.any(prices <= 0)
        has_nan = np.any(np.isnan(prices))
        has_inf = np.any(np.isinf(prices))

        # Determine overall validity
        is_valid = not (has_negative or has_nan or has_inf)

        # Calculate statistics (using all values, even if some are invalid)
        stats = {
            "min": float(np.min(prices)),
            "max": float(np.max(prices)),
            "mean": float(np.mean(prices)),
            "std": float(np.std(prices)),
            "median": float(np.median(prices)),
            "q25": float(np.percentile(prices, 25)),
            "q75": float(np.percentile(prices, 75))
        }

        # Calculate price range information
        price_range_val = stats["max"] - stats["min"]
        if stats["min"] > 0:
            percent_change = (stats["max"] / stats["min"] - 1) * 100
        else:
            percent_change = float('nan')

        range_info = {
            "range": price_range_val,
            "percent_change": percent_change,
            "first_price": float(prices[0]),
            "last_price": float(prices[-1])
        }

        # Generate warnings
        warnings = []
        if has_negative:
            warnings.append("Contains negative prices")
        if has_nan:
            warnings.append("Contains NaN values")
        if has_inf:
            warnings.append("Contains infinity values")

        # Check for extreme volatility
        if stats["std"] > 2 * stats["mean"] and stats["mean"] > 0:
            warnings.append("Prices show extreme volatility")

        # Generate informational messages
        messages = []
        if is_valid:
            messages.append(f"All prices valid: range ${stats['min']:.2f} - ${stats['max']:.2f}")
        else:
            messages.append("Price validation FAILED - see warnings")

        messages.append(
            f"Statistics: mean=${stats['mean']:.2f}, std=${stats['std']:.2f}, "
            f"median=${stats['median']:.2f}"
        )

        if has_negative:
            messages.append("ACTION REQUIRED: Negative prices detected - may indicate forecast issues")
        if has_nan or has_inf:
            messages.append("ACTION REQUIRED: NaN or inf values - invalid for trading/analysis")

        logger.info(
            f"Price validation complete: is_valid={is_valid}, "
            f"has_negative={has_negative}, has_nan={has_nan}, has_inf={has_inf}"
        )

        # Build and return validation report
        report = {
            "is_valid": is_valid,
            "has_negative_prices": has_negative,
            "has_nan_values": has_nan,
            "has_inf_values": has_inf,
            "statistics": stats,
            "price_range": range_info,
            "warnings": warnings,
            "messages": messages
        }

        for message in messages:
            if "FAILED" in message:
                logger.warning(message)
            else:
                logger.info(message)

        return report

    except (ValueError, TypeError) as e:
        error_msg = f"Input validation failed in price validation: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise
    except Exception as e:
        error_msg = f"Failed to validate prices: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise
