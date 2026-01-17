"""
Hybrid Combiner Module for Hybrid LSTM-ARIMA Forecasting System

This module handles the combination of ARIMA linear forecasts with LSTM residual
forecasts to produce final hybrid predictions in RETURNS SPACE. It implements the
combination formula:

R̂_t = L̂_t + N̂_t

Where:
- R̂_t = Final hybrid returns forecast (in RETURNS SPACE)
- L̂_t = ARIMA linear prediction (in RETURNS SPACE)
- N̂_t = LSTM residual prediction (in RETURNS SPACE)

All predictions are in RETURNS SPACE (percentage changes).

Functions:
    - combine_forecasts: Combine linear and non-linear components in returns space
    - get_hybrid_component_breakdown: Extract component statistics and contributions
    - combine_predictions: DEPRECATED - use combine_forecasts()
    - get_component_details: DEPRECATED - use get_hybrid_component_breakdown()

Error Handling & Logging: Section 10, Error Handling Strategy
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any

from src.exceptions import DataValidationError
from src.logger_config import get_logger, log_exception


# Configure module logging
logger = get_logger(__name__)


def combine_forecasts(
    arima_forecast: np.ndarray,
    lstm_forecast: np.ndarray
) -> Dict[str, Any]:
    """
    Combine ARIMA linear forecasts with LSTM residual forecasts (returns space).

    Implements the hybrid forecasting formula R̂_t = L̂_t + N̂_t where:
    - R̂_t = Final hybrid returns forecast (in RETURNS SPACE)
    - L̂_t = ARIMA linear prediction (in RETURNS SPACE)
    - N̂_t = LSTM residual prediction (in RETURNS SPACE)

    Both inputs and output are in RETURNS SPACE (percentage changes).

    Args:
        arima_forecast (np.ndarray): ARIMA linear predictions array.
            Shape: (n_steps,) - 1D array of ARIMA forecast values (returns space)
        lstm_forecast (np.ndarray): LSTM residual predictions array.
            Shape: (m_steps,) - 1D array of LSTM residual forecast values (returns space)
            Note: May differ in length from arima_forecast due to LSTM window preprocessing

    Returns:
        Dict[str, Any]: Standardized result dictionary containing:
            - predictions (np.ndarray): Combined hybrid forecast array (returns space)
                Shape: (aligned_steps,) - Same length as shorter input array
            - arima_component (np.ndarray): Linear ARIMA component (L̂_t)
                Shape: (aligned_steps,)
            - lstm_component (np.ndarray): Non-linear LSTM component (N̂_t)
                Shape: (aligned_steps,)
            - component_stats (dict): Statistics on each component contribution:
                - arima_mean (float): Mean of ARIMA component
                - arima_std (float): Standard deviation of ARIMA component
                - lstm_mean (float): Mean of LSTM component
                - lstm_std (float): Standard deviation of LSTM component
                - arima_contribution (float): Percentage contribution of ARIMA (0-100)
                - lstm_contribution (float): Percentage contribution of LSTM (0-100)
            - alignment_info (dict): Information about array alignment:
                - original_arima_length (int): Original ARIMA forecast length
                - original_lstm_length (int): Original LSTM forecast length
                - aligned_length (int): Final aligned length used

    Raises:
        ValueError: If input arrays are empty, have incompatible shapes,
            contain NaN/inf values, or have zero or negative dimensions
        TypeError: If inputs are not numpy arrays or not numeric types

    Examples:
        >>> arima_pred = np.array([0.01, 0.02, 0.015])  # returns space
        >>> lstm_pred = np.array([0.002, -0.001, 0.003])  # returns space
        >>> result = combine_forecasts(arima_pred, lstm_pred)
        >>> print(result['predictions'])
        [0.012, 0.019, 0.018]
        >>> print(result['component_stats']['arima_contribution'])
        95.5

    Note:
        - Predictions are aligned to the length of the shorter array
        - Component contributions are calculated as percentage of total variance
        - All input arrays must be 1D numeric arrays
        - NaN and inf values are not allowed in input arrays
        - All values are in RETURNS SPACE (percentage changes)
    """
    # Input validation - type checks
    if not isinstance(arima_forecast, np.ndarray):
        error_msg = f"arima_forecast must be numpy array, got {type(arima_forecast)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if not isinstance(lstm_forecast, np.ndarray):
        error_msg = f"lstm_forecast must be numpy array, got {type(lstm_forecast)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Input validation - numeric type checks
    if not np.issubdtype(arima_forecast.dtype, np.number):
        error_msg = f"arima_forecast must contain numeric values, got dtype {arima_forecast.dtype}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if not np.issubdtype(lstm_forecast.dtype, np.number):
        error_msg = f"lstm_forecast must contain numeric values, got dtype {lstm_forecast.dtype}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Input validation - dimensionality checks
    if arima_forecast.ndim != 1:
        error_msg = f"arima_forecast must be 1D array, got shape {arima_forecast.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if lstm_forecast.ndim != 1:
        error_msg = f"lstm_forecast must be 1D array, got shape {lstm_forecast.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Input validation - empty checks
    if len(arima_forecast) == 0:
        error_msg = "arima_forecast is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(lstm_forecast) == 0:
        error_msg = "lstm_forecast is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Input validation - NaN/inf checks
    if np.isnan(arima_forecast).any() or np.isinf(arima_forecast).any():
        error_msg = "arima_forecast contains NaN or inf values"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if np.isnan(lstm_forecast).any() or np.isinf(lstm_forecast).any():
        error_msg = "lstm_forecast contains NaN or inf values"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.info(
            f"Combining predictions: arima_shape={arima_forecast.shape}, "
            f"lstm_shape={lstm_forecast.shape}"
        )

        original_arima_length = len(arima_forecast)
        original_lstm_length = len(lstm_forecast)

        # Align predictions to shorter length to avoid index mismatch
        # This handles cases where ARIMA and LSTM processing produce different lengths
        aligned_length = min(original_arima_length, original_lstm_length)
        logger.debug(
            f"Aligning predictions: original_arima={original_arima_length}, "
            f"original_lstm={original_lstm_length}, aligned_length={aligned_length}"
        )

        # Trim arrays to aligned length
        arima_aligned = arima_forecast[:aligned_length]
        lstm_aligned = lstm_forecast[:aligned_length]

        # Combine predictions using hybrid formula: ŷ_t = L̂_t + N̂_t
        combined_predictions = arima_aligned + lstm_aligned

        logger.debug(
            f"Hybrid combination completed: "
            f"combined_mean={combined_predictions.mean():.6f}, "
            f"combined_std={combined_predictions.std():.6f}, "
            f"combined_min={combined_predictions.min():.6f}, "
            f"combined_max={combined_predictions.max():.6f}"
        )

        # Calculate component statistics
        arima_mean = float(np.mean(arima_aligned))
        arima_std = float(np.std(arima_aligned))
        lstm_mean = float(np.mean(lstm_aligned))
        lstm_std = float(np.std(lstm_aligned))

        # Calculate contribution percentages based on variance
        arima_variance = np.var(arima_aligned)
        lstm_variance = np.var(lstm_aligned)
        total_variance = arima_variance + lstm_variance

        if total_variance > 0:
            arima_contribution = float((arima_variance / total_variance) * 100)
            lstm_contribution = float((lstm_variance / total_variance) * 100)
        else:
            # If both have zero variance, attribute 50-50
            arima_contribution = 50.0
            lstm_contribution = 50.0

        logger.info(
            f"Component statistics calculated: "
            f"arima_mean={arima_mean:.6f}, lstm_mean={lstm_mean:.6f}, "
            f"arima_contribution={arima_contribution:.2f}%, "
            f"lstm_contribution={lstm_contribution:.2f}%"
        )

        # Build component statistics dictionary
        component_stats = {
            "arima_mean": arima_mean,
            "arima_std": arima_std,
            "lstm_mean": lstm_mean,
            "lstm_std": lstm_std,
            "arima_contribution": arima_contribution,
            "lstm_contribution": lstm_contribution
        }

        # Build alignment information dictionary
        alignment_info = {
            "original_arima_length": original_arima_length,
            "original_lstm_length": original_lstm_length,
            "aligned_length": aligned_length
        }

        # Build and return result dictionary
        result = {
            "predictions": combined_predictions,
            "arima_component": arima_aligned,
            "lstm_component": lstm_aligned,
            "component_stats": component_stats,
            "alignment_info": alignment_info
        }

        logger.info(f"Hybrid combination successful. Final predictions shape: {combined_predictions.shape}")

        return result

    except (ValueError, TypeError) as e:
        error_msg = f"Input validation failed during prediction combination: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise DataValidationError(
            error_message=error_msg,
            data_shape=(len(arima_forecast), len(lstm_forecast))
        ) from e
    except Exception as e:
        error_msg = f"Failed to combine predictions: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise


def get_hybrid_component_breakdown(
    arima_component: np.ndarray,
    lstm_component: np.ndarray
) -> Dict[str, Any]:
    """
    Extract detailed breakdown of component contributions to the hybrid forecast.

    Analyzes the ARIMA linear and LSTM non-linear components to provide
    comprehensive statistics on their individual contributions, relationships,
    and impact on the final hybrid forecast (R̂_t = L̂_t + N̂_t).

    Args:
        arima_component (np.ndarray): ARIMA linear predictions array.
            Shape: (n_steps,) - 1D array of already-aligned ARIMA values
        lstm_component (np.ndarray): LSTM residual predictions array.
            Shape: (n_steps,) - 1D array of already-aligned LSTM residual values
            Must have same length as arima_component

    Returns:
        Dict[str, Any]: Comprehensive component statistics dictionary containing:
            - arima (dict): ARIMA component statistics:
                - mean (float): Mean value of ARIMA component
                - std (float): Standard deviation of ARIMA component
                - min (float): Minimum value in ARIMA component
                - max (float): Maximum value in ARIMA component
                - range (float): Max - Min of ARIMA component
                - variance (float): Variance of ARIMA component
            - lstm (dict): LSTM component statistics:
                - mean (float): Mean value of LSTM component
                - std (float): Standard deviation of LSTM component
                - min (float): Minimum value in LSTM component
                - max (float): Maximum value in LSTM component
                - range (float): Max - Min of LSTM component
                - variance (float): Variance of LSTM component
            - combined (dict): Combined hybrid statistics:
                - mean (float): Mean of hybrid forecast
                - std (float): Standard deviation of hybrid forecast
                - min (float): Minimum in hybrid forecast
                - max (float): Maximum in hybrid forecast
                - range (float): Max - Min of hybrid forecast
            - correlation (dict): Relationship between components:
                - pearson_correlation (float): Pearson correlation coefficient between components
                    Range: [-1, 1] where 1 = perfect positive, -1 = perfect negative, 0 = independent
                - covariance (float): Covariance between components
            - contributions (dict): Percentage contributions to total variance:
                - arima_variance_percent (float): ARIMA variance as percent of total
                - lstm_variance_percent (float): LSTM variance as percent of total
                - combined_variance (float): Total variance of hybrid forecast
            - lengths (dict): Information about array dimensions:
                - arima_length (int): Length of ARIMA component
                - lstm_length (int): Length of LSTM component
                - components_aligned (bool): True if components have same length

    Raises:
        ValueError: If input arrays are empty, have mismatched lengths,
            or contain NaN/inf values
        TypeError: If inputs are not numpy arrays or not numeric types

    Examples:
        >>> arima_comp = np.array([100.5, 101.2, 102.1])
        >>> lstm_comp = np.array([0.5, -0.3, 0.2])
        >>> details = get_component_details(arima_comp, lstm_comp)
        >>> print(f"ARIMA mean: {details['arima']['mean']:.2f}")
        >>> print(f"Correlation: {details['correlation']['pearson_correlation']:.4f}")
        >>> print(f"ARIMA contribution: {details['contributions']['arima_variance_percent']:.2f}%")

    Note:
        - Components should already be aligned to same length before calling
        - If components have different lengths, function returns error
        - Correlation coefficient calculated using Pearson method
        - All statistics calculated on the provided component values directly
        - NaN and inf values are not allowed in input arrays
    """
    # Input validation - type checks
    if not isinstance(arima_component, np.ndarray):
        error_msg = f"arima_component must be numpy array, got {type(arima_component)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if not isinstance(lstm_component, np.ndarray):
        error_msg = f"lstm_component must be numpy array, got {type(lstm_component)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Input validation - numeric type checks
    if not np.issubdtype(arima_component.dtype, np.number):
        error_msg = f"arima_component must contain numeric values, got dtype {arima_component.dtype}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if not np.issubdtype(lstm_component.dtype, np.number):
        error_msg = f"lstm_component must contain numeric values, got dtype {lstm_component.dtype}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Input validation - dimensionality checks
    if arima_component.ndim != 1:
        error_msg = f"arima_component must be 1D array, got shape {arima_component.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if lstm_component.ndim != 1:
        error_msg = f"lstm_component must be 1D array, got shape {lstm_component.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Input validation - empty checks
    if len(arima_component) == 0:
        error_msg = "arima_component is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(lstm_component) == 0:
        error_msg = "lstm_component is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Input validation - length matching
    if len(arima_component) != len(lstm_component):
        error_msg = (
            f"Components must have same length. "
            f"arima_component: {len(arima_component)}, lstm_component: {len(lstm_component)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Input validation - NaN/inf checks
    if np.isnan(arima_component).any() or np.isinf(arima_component).any():
        error_msg = "arima_component contains NaN or inf values"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if np.isnan(lstm_component).any() or np.isinf(lstm_component).any():
        error_msg = "lstm_component contains NaN or inf values"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.info(
            f"Extracting component details: arima_shape={arima_component.shape}, "
            f"lstm_shape={lstm_component.shape}"
        )

        # Calculate ARIMA component statistics
        arima_mean = float(np.mean(arima_component))
        arima_std = float(np.std(arima_component))
        arima_min = float(np.min(arima_component))
        arima_max = float(np.max(arima_component))
        arima_range = arima_max - arima_min
        arima_variance = float(np.var(arima_component))

        arima_stats = {
            "mean": arima_mean,
            "std": arima_std,
            "min": arima_min,
            "max": arima_max,
            "range": arima_range,
            "variance": arima_variance
        }

        # Calculate LSTM component statistics
        lstm_mean = float(np.mean(lstm_component))
        lstm_std = float(np.std(lstm_component))
        lstm_min = float(np.min(lstm_component))
        lstm_max = float(np.max(lstm_component))
        lstm_range = lstm_max - lstm_min
        lstm_variance = float(np.var(lstm_component))

        lstm_stats = {
            "mean": lstm_mean,
            "std": lstm_std,
            "min": lstm_min,
            "max": lstm_max,
            "range": lstm_range,
            "variance": lstm_variance
        }

        # Calculate combined (hybrid) statistics
        combined = arima_component + lstm_component
        combined_mean = float(np.mean(combined))
        combined_std = float(np.std(combined))
        combined_min = float(np.min(combined))
        combined_max = float(np.max(combined))
        combined_range = combined_max - combined_min

        combined_stats = {
            "mean": combined_mean,
            "std": combined_std,
            "min": combined_min,
            "max": combined_max,
            "range": combined_range
        }

        # Calculate correlation and covariance between components
        pearson_correlation = float(np.corrcoef(arima_component, lstm_component)[0, 1])
        covariance = float(np.cov(arima_component, lstm_component)[0, 1])

        correlation_stats = {
            "pearson_correlation": pearson_correlation,
            "covariance": covariance
        }

        # Calculate contribution percentages based on variance
        total_variance = arima_variance + lstm_variance

        if total_variance > 0:
            arima_variance_percent = float((arima_variance / total_variance) * 100)
            lstm_variance_percent = float((lstm_variance / total_variance) * 100)
        else:
            # If both have zero variance, attribute 50-50
            arima_variance_percent = 50.0
            lstm_variance_percent = 50.0

        combined_variance = float(np.var(combined))

        contribution_stats = {
            "arima_variance_percent": arima_variance_percent,
            "lstm_variance_percent": lstm_variance_percent,
            "combined_variance": combined_variance
        }

        # Build length information dictionary
        length_info = {
            "arima_length": len(arima_component),
            "lstm_length": len(lstm_component),
            "components_aligned": len(arima_component) == len(lstm_component)
        }

        logger.info(
            f"Component details extracted: "
            f"arima_mean={arima_mean:.6f}, lstm_mean={lstm_mean:.6f}, "
            f"correlation={pearson_correlation:.6f}, "
            f"arima_var%={arima_variance_percent:.2f}%, lstm_var%={lstm_variance_percent:.2f}%"
        )

        # Build and return result dictionary
        result = {
            "arima": arima_stats,
            "lstm": lstm_stats,
            "combined": combined_stats,
            "correlation": correlation_stats,
            "contributions": contribution_stats,
            "lengths": length_info
        }

        logger.info("Component details extraction successful")

        return result

    except (ValueError, TypeError) as e:
        error_msg = f"Input validation failed during component detail extraction: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise DataValidationError(
            error_message=error_msg,
            data_shape=(len(arima_component), len(lstm_component))
        ) from e
    except Exception as e:
        error_msg = f"Failed to extract component details: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# ============================================================================


def combine_predictions(
    arima_forecast: np.ndarray,
    lstm_residual_forecast: np.ndarray
) -> Dict[str, Any]:
    """
    DEPRECATED: Use combine_forecasts() instead.

    This function is a backward-compatibility wrapper for combine_forecasts().
    It maintains the old function name and parameter names for existing code.

    See combine_forecasts() for full documentation.
    """
    logger.warning(
        "combine_predictions() is deprecated. Use combine_forecasts() instead. "
        "Parameter 'lstm_residual_forecast' has been renamed to 'lstm_forecast'."
    )
    return combine_forecasts(arima_forecast, lstm_residual_forecast)


def get_component_details(
    arima_component: np.ndarray,
    lstm_component: np.ndarray
) -> Dict[str, Any]:
    """
    DEPRECATED: Use get_hybrid_component_breakdown() instead.

    This function is a backward-compatibility wrapper for get_hybrid_component_breakdown().
    It maintains the old function name for existing code.

    See get_hybrid_component_breakdown() for full documentation.
    """
    logger.warning(
        "get_component_details() is deprecated. Use get_hybrid_component_breakdown() instead."
    )
    return get_hybrid_component_breakdown(arima_component, lstm_component)
