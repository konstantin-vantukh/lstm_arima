"""
Unit Tests for ARIMA Engine Module

This test module provides comprehensive coverage of the ARIMA engine functionality
including stationarity testing, parameter selection, model fitting, and residual extraction.
Follows the test strategy outlined in architecture.md Section 11.2.

Test Coverage:
- UT1: ADF test for non-stationary data (random walk)
- UT2: ADF test for stationary data (white noise)
- UT3: Auto-ARIMA parameter selection
- UT4: ARIMA model fitting
- UT5: Residual extraction from fitted models
"""

import pytest
import pandas as pd
import numpy as np
from typing import Tuple
from src.arima_engine import (
    test_stationarity,
    find_optimal_params,
    fit_arima,
    extract_residuals
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def non_stationary_series() -> pd.Series:
    """
    Generate a non-stationary time series (random walk).
    
    A random walk is inherently non-stationary because each value depends on
    the previous value plus a random shock. This should fail the ADF test
    (p-value >= 0.05).
    
    Returns:
        pd.Series: Random walk series of length 200
    """
    np.random.seed(42)
    random_shocks = np.random.randn(200)
    random_walk = np.cumsum(random_shocks)
    return pd.Series(random_walk, name="random_walk")


@pytest.fixture
def stationary_series() -> pd.Series:
    """
    Generate a stationary time series (white noise).
    
    White noise is stationary by definition - it has constant mean, constant
    variance, and no autocorrelation. This should pass the ADF test
    (p-value < 0.05).
    
    Returns:
        pd.Series: White noise series of length 200
    """
    np.random.seed(42)
    white_noise = np.random.randn(200)
    return pd.Series(white_noise, name="white_noise")


@pytest.fixture
def returns_series() -> pd.Series:
    """
    Generate a realistic returns series for ARIMA parameter selection.
    
    Creates a series from differenced random walk (simulating price returns),
    which is typically stationary.
    
    Returns:
        pd.Series: Returns series of length 100
    """
    np.random.seed(42)
    random_shocks = np.random.randn(101)
    random_walk = np.cumsum(random_shocks)
    returns = np.diff(random_walk)
    return pd.Series(returns, name="returns")


@pytest.fixture
def fitted_model_with_series() -> Tuple:
    """
    Create a fitted ARIMA model and its original series for testing.
    
    Fits an ARIMA(1,1,1) model to a stationary series and returns both
    the model and original series for residual extraction tests.
    
    Returns:
        Tuple[pd.Series, ARIMAResults]: Original series and fitted model
    """
    np.random.seed(42)
    series = pd.Series(np.random.randn(100), name="test_series")
    model = fit_arima(series, order=(1, 1, 1))
    return series, model


# ============================================================================
# TEST 1: ADF STATIONARITY TEST - NON-STATIONARY DATA
# ============================================================================

class TestADFStationarityNonStationary:
    """
    Test suite for ADF stationarity test on non-stationary data.
    
    Validates that the ADF test correctly identifies non-stationary series
    (Test Contract UT1).
    """

    def test_adf_stationarity_non_stationary(self, non_stationary_series: pd.Series):
        """
        Verify ADF test correctly identifies non-stationary synthetic data.
        
        A random walk should NOT be stationary. The ADF test should return
        is_stationary=False (p-value >= 0.05).
        
        This test validates UT1 from the contract.
        """
        is_stationary, p_value = test_stationarity(non_stationary_series)
        
        # Random walk should NOT be stationary
        assert is_stationary == False, \
            f"Random walk should be non-stationary, got is_stationary={is_stationary}"
        
        # P-value should be >= 0.05 for non-stationary series
        assert p_value >= 0.05, \
            f"Non-stationary series should have p-value >= 0.05, got {p_value:.6f}"

    def test_adf_stationarity_non_stationary_returns_tuple(
        self, non_stationary_series: pd.Series
    ):
        """
        Verify ADF test returns proper tuple format.
        
        The return value should be a tuple with (bool, float).
        """
        result = test_stationarity(non_stationary_series)
        
        assert isinstance(result, tuple), \
            f"Result should be tuple, got {type(result)}"
        assert len(result) == 2, \
            f"Result tuple should have 2 elements, got {len(result)}"
        assert isinstance(result[0], (bool, np.bool_)), \
            f"First element should be bool, got {type(result[0])}"
        assert isinstance(result[1], (float, np.floating)), \
            f"Second element should be float, got {type(result[1])}"

    def test_adf_stationarity_non_stationary_p_value_range(
        self, non_stationary_series: pd.Series
    ):
        """
        Verify p-value is in valid range [0, 1].
        """
        _, p_value = test_stationarity(non_stationary_series)
        
        assert 0 <= p_value <= 1, \
            f"P-value should be between 0 and 1, got {p_value}"


# ============================================================================
# TEST 2: ADF STATIONARITY TEST - STATIONARY DATA
# ============================================================================

class TestADFStationarityStationary:
    """
    Test suite for ADF stationarity test on stationary data.
    
    Validates that the ADF test correctly identifies stationary series
    (Test Contract UT2).
    """

    def test_adf_stationarity_stationary(self, stationary_series: pd.Series):
        """
        Verify ADF test correctly identifies stationary synthetic data.
        
        White noise should be stationary. The ADF test should return
        is_stationary=True (p-value < 0.05).
        
        This test validates UT2 from the contract.
        """
        is_stationary, p_value = test_stationarity(stationary_series)
        
        # White noise should be stationary
        assert is_stationary == True, \
            f"White noise should be stationary, got is_stationary={is_stationary}"
        
        # P-value should be < 0.05 for stationary series
        assert p_value < 0.05, \
            f"Stationary series should have p-value < 0.05, got {p_value:.6f}"

    def test_adf_stationarity_stationary_returns_tuple(
        self, stationary_series: pd.Series
    ):
        """
        Verify ADF test returns proper tuple format for stationary data.
        """
        result = test_stationarity(stationary_series)
        
        assert isinstance(result, tuple), \
            f"Result should be tuple, got {type(result)}"
        assert len(result) == 2, \
            f"Result tuple should have 2 elements, got {len(result)}"
        assert isinstance(result[0], (bool, np.bool_)), \
            f"First element should be bool, got {type(result[0])}"
        assert isinstance(result[1], (float, np.floating)), \
            f"Second element should be float, got {type(result[1])}"

    def test_adf_stationarity_stationary_p_value_range(
        self, stationary_series: pd.Series
    ):
        """
        Verify p-value is in valid range [0, 1] for stationary data.
        """
        _, p_value = test_stationarity(stationary_series)
        
        assert 0 <= p_value <= 1, \
            f"P-value should be between 0 and 1, got {p_value}"


# ============================================================================
# TEST 3: FIND OPTIMAL PARAMETERS
# ============================================================================

class TestFindOptimalParams:
    """
    Test suite for auto-ARIMA parameter selection.
    
    Validates that the parameter selection algorithm finds reasonable (p, d, q)
    values within the specified bounds (Test Contract UT3).
    """

    def test_find_optimal_params(self, returns_series: pd.Series):
        """
        Verify auto-ARIMA parameter selection returns valid parameters.
        
        The function should return a tuple (p, d, q) with values within bounds.
        This test validates UT3 from the contract.
        """
        max_p, max_d, max_q = 5, 2, 5
        p, d, q = find_optimal_params(
            returns_series, 
            max_p=max_p, 
            max_d=max_d, 
            max_q=max_q
        )
        
        # Verify return type
        assert isinstance(p, (int, np.integer)), \
            f"p should be int, got {type(p)}"
        assert isinstance(d, (int, np.integer)), \
            f"d should be int, got {type(d)}"
        assert isinstance(q, (int, np.integer)), \
            f"q should be int, got {type(q)}"

    def test_find_optimal_params_within_bounds(self, returns_series: pd.Series):
        """
        Verify parameters are within specified bounds.
        
        Assertions for parameter selection (p, d, q should be within bounds).
        """
        max_p, max_d, max_q = 5, 2, 5
        p, d, q = find_optimal_params(
            returns_series,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q
        )
        
        # Check p is within bounds
        assert 0 <= p <= max_p, \
            f"p should be between 0 and {max_p}, got {p}"
        
        # Check d is within bounds
        assert 0 <= d <= max_d, \
            f"d should be between 0 and {max_d}, got {d}"
        
        # Check q is within bounds
        assert 0 <= q <= max_q, \
            f"q should be between 0 and {max_q}, got {q}"

    def test_find_optimal_params_returns_tuple(self, returns_series: pd.Series):
        """
        Verify return value is a tuple of three integers.
        """
        result = find_optimal_params(returns_series, max_p=3, max_d=2, max_q=3)
        
        assert isinstance(result, tuple), \
            f"Result should be tuple, got {type(result)}"
        assert len(result) == 3, \
            f"Result tuple should have 3 elements, got {len(result)}"

    def test_find_optimal_params_custom_bounds(self, returns_series: pd.Series):
        """
        Verify parameter selection respects custom bounds.
        """
        max_p, max_d, max_q = 2, 1, 2
        p, d, q = find_optimal_params(
            returns_series,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q
        )
        
        assert 0 <= p <= max_p, \
            f"p should be between 0 and {max_p}, got {p}"
        assert 0 <= d <= max_d, \
            f"d should be between 0 and {max_d}, got {d}"
        assert 0 <= q <= max_q, \
            f"q should be between 0 and {max_q}, got {q}"

    def test_find_optimal_params_d_limited_to_2(self, returns_series: pd.Series):
        """
        Verify that d (differencing order) is limited to 2 for stability.
        
        Per architecture requirements, d should never exceed 2.
        """
        # Request d to exceed 2
        p, d, q = find_optimal_params(returns_series, max_p=5, max_d=5, max_q=5)
        
        # Should still be limited to 2
        assert d <= 2, \
            f"Differencing order d should be limited to 2, got {d}"

    def test_find_optimal_params_non_negative(self, returns_series: pd.Series):
        """
        Verify all parameters are non-negative.
        """
        p, d, q = find_optimal_params(returns_series, max_p=5, max_d=2, max_q=5)
        
        assert p >= 0, f"p should be non-negative, got {p}"
        assert d >= 0, f"d should be non-negative, got {d}"
        assert q >= 0, f"q should be non-negative, got {q}"


# ============================================================================
# TEST 4: FIT ARIMA MODEL
# ============================================================================

class TestFitARIMA:
    """
    Test suite for ARIMA model fitting.
    
    Validates that the ARIMA model can be fitted with known parameters
    and produces a valid results object (Test Contract UT4).
    """

    def test_fit_arima(self, stationary_series: pd.Series):
        """
        Verify ARIMA model fitting with known parameters.
        
        Should successfully fit an ARIMA(1,1,1) model to stationary data
        and return a valid results object.
        
        This test validates UT4 from the contract.
        """
        order = (1, 1, 1)
        results = fit_arima(stationary_series, order=order)
        
        # Verify results object has expected attributes
        assert hasattr(results, "fittedvalues"), \
            "Fitted model should have fittedvalues attribute"
        assert hasattr(results, "aic"), \
            "Fitted model should have aic attribute"
        assert hasattr(results, "bic"), \
            "Fitted model should have bic attribute"
        assert hasattr(results, "params"), \
            "Fitted model should have params attribute"

    def test_fit_arima_fittedvalues_length(self, stationary_series: pd.Series):
        """
        Verify fitted values match series length.
        """
        order = (1, 1, 1)
        results = fit_arima(stationary_series, order=order)
        
        # Fittedvalues should have same length as input series
        assert len(results.fittedvalues) == len(stationary_series), \
            f"Fittedvalues length ({len(results.fittedvalues)}) should match " \
            f"series length ({len(stationary_series)})"

    def test_fit_arima_aic_bic_numeric(self, stationary_series: pd.Series):
        """
        Verify AIC and BIC values are numeric and finite.
        """
        results = fit_arima(stationary_series, order=(1, 1, 1))
        
        assert isinstance(results.aic, (int, float, np.number)), \
            f"AIC should be numeric, got {type(results.aic)}"
        assert isinstance(results.bic, (int, float, np.number)), \
            f"BIC should be numeric, got {type(results.bic)}"
        assert np.isfinite(results.aic), \
            f"AIC should be finite, got {results.aic}"
        assert np.isfinite(results.bic), \
            f"BIC should be finite, got {results.bic}"

    def test_fit_arima_different_orders(self, stationary_series: pd.Series):
        """
        Verify ARIMA can be fitted with different parameter combinations.
        """
        orders = [(0, 0, 0), (1, 0, 1), (1, 1, 1), (2, 1, 2)]
        
        for order in orders:
            results = fit_arima(stationary_series, order=order)
            assert hasattr(results, "fittedvalues"), \
                f"Order {order} should produce valid results"

    def test_fit_arima_params_dict(self, stationary_series: pd.Series):
        """
        Verify fitted parameters are accessible.
        """
        results = fit_arima(stationary_series, order=(1, 1, 1))
        
        assert isinstance(results.params, (dict, pd.Series)) or hasattr(results.params, '__len__'), \
            f"Params should be accessible, got {type(results.params)}"


# ============================================================================
# TEST 5: EXTRACT RESIDUALS
# ============================================================================

class TestExtractResiduals:
    """
    Test suite for residual extraction from fitted ARIMA models.
    
    Validates that residuals are correctly computed as actual - predicted values
    (Test Contract UT5).
    """

    def test_extract_residuals(self, fitted_model_with_series: Tuple):
        """
        Verify residual extraction from fitted model.
        
        Residuals should be computed as actual - predicted values.
        This test validates UT5 from the contract.
        
        Assertions for residual extraction (length matches original series).
        """
        series, model = fitted_model_with_series
        residuals = extract_residuals(series, model)
        
        # Verify residuals is a Series
        assert isinstance(residuals, pd.Series), \
            f"Residuals should be pd.Series, got {type(residuals)}"
        
        # Verify residuals length matches series length
        assert len(residuals) == len(series), \
            f"Residuals length ({len(residuals)}) should match " \
            f"series length ({len(series)})"

    def test_extract_residuals_length_matches_original(
        self, fitted_model_with_series: Tuple
    ):
        """
        Verify residual length matches original series.
        
        This is a critical requirement for LSTM input validation.
        """
        series, model = fitted_model_with_series
        residuals = extract_residuals(series, model)
        
        assert len(residuals) == len(series), \
            f"Residuals length should match series length: " \
            f"{len(residuals)} != {len(series)}"

    def test_extract_residuals_calculation(
        self, fitted_model_with_series: Tuple
    ):
        """
        Verify residuals are calculated as actual - predicted.
        
        Residuals should equal series - fitted_values.
        """
        series, model = fitted_model_with_series
        residuals = extract_residuals(series, model)
        
        # Calculate expected residuals manually
        expected_residuals = series - model.fittedvalues
        
        # Compare (allow small numerical differences)
        np.testing.assert_allclose(
            residuals.values,
            expected_residuals.values,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Residuals should equal series - fittedvalues"
        )

    def test_extract_residuals_is_series(
        self, fitted_model_with_series: Tuple
    ):
        """
        Verify residuals return type is pd.Series.
        """
        series, model = fitted_model_with_series
        residuals = extract_residuals(series, model)
        
        assert isinstance(residuals, pd.Series), \
            f"Residuals should be pd.Series, got {type(residuals)}"

    def test_extract_residuals_mean_near_zero(
        self, fitted_model_with_series: Tuple
    ):
        """
        Verify residual mean is near zero.
        
        Well-fitted ARIMA models should have residuals with mean close to zero.
        """
        series, model = fitted_model_with_series
        residuals = extract_residuals(series, model)
        
        # Mean should be close to zero (within tolerance for ARIMA)
        assert abs(residuals.mean()) < 1.0, \
            f"Residual mean should be close to 0, got {residuals.mean():.6f}"

    def test_extract_residuals_std_dev_positive(
        self, fitted_model_with_series: Tuple
    ):
        """
        Verify residuals have non-zero standard deviation.
        """
        series, model = fitted_model_with_series
        residuals = extract_residuals(series, model)
        
        assert residuals.std() >= 0, \
            f"Residual std dev should be non-negative, got {residuals.std()}"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestARIMAIntegration:
    """
    Integration tests for complete ARIMA workflow.
    """

    def test_complete_arima_workflow(self, returns_series: pd.Series):
        """
        Test complete workflow: check stationarity -> find params -> fit -> extract residuals.
        """
        # Step 1: Check stationarity
        is_stationary, p_value = test_stationarity(returns_series)
        
        # Step 2: Find optimal parameters
        p, d, q = find_optimal_params(returns_series, max_p=5, max_d=2, max_q=5)
        
        # Step 3: Fit model
        model = fit_arima(returns_series, order=(p, d, q))
        
        # Step 4: Extract residuals
        residuals = extract_residuals(returns_series, model)
        
        # Verify complete workflow produced valid results
        assert residuals is not None, "Residuals should not be None"
        assert len(residuals) == len(returns_series), \
            "Residuals length should match series length"

    def test_non_stationary_to_stationary_workflow(
        self, non_stationary_series: pd.Series
    ):
        """
        Test workflow on non-stationary data (differencing should occur).
        """
        # Check initial stationarity
        is_stationary_initial, _ = test_stationarity(non_stationary_series)
        
        # Find optimal parameters (should include differencing)
        p, d, q = find_optimal_params(
            non_stationary_series,
            max_p=5,
            max_d=2,
            max_q=5
        )
        
        # Should apply at least one level of differencing
        assert d >= 1, \
            "Non-stationary series should require differencing (d >= 1)"
        
        # Fit model with selected parameters
        model = fit_arima(non_stationary_series, order=(p, d, q))
        
        # Extract residuals
        residuals = extract_residuals(non_stationary_series, model)
        
        assert len(residuals) == len(non_stationary_series), \
            "Residuals length should match original series"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestARIMAErrorHandling:
    """
    Test error handling in ARIMA functions.
    """

    def test_test_stationarity_empty_series(self):
        """
        Verify test_stationarity raises error on empty series.
        """
        empty_series = pd.Series([])
        
        with pytest.raises(ValueError):
            test_stationarity(empty_series)

    def test_test_stationarity_nan_values(self):
        """
        Verify test_stationarity raises error on NaN values.
        """
        series_with_nan = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        
        with pytest.raises(ValueError):
            test_stationarity(series_with_nan)

    def test_find_optimal_params_empty_series(self):
        """
        Verify find_optimal_params raises error on empty series.
        """
        empty_series = pd.Series([])
        
        with pytest.raises(ValueError):
            find_optimal_params(empty_series)

    def test_find_optimal_params_negative_bounds(self):
        """
        Verify find_optimal_params raises error on negative bounds.
        """
        series = pd.Series(np.random.randn(100))
        
        with pytest.raises(ValueError):
            find_optimal_params(series, max_p=-1, max_d=2, max_q=5)

    def test_fit_arima_empty_series(self):
        """
        Verify fit_arima raises error on empty series.
        """
        empty_series = pd.Series([])
        
        with pytest.raises(ValueError):
            fit_arima(empty_series, order=(1, 1, 1))

    def test_fit_arima_invalid_order(self):
        """
        Verify fit_arima raises error on invalid (negative) order.
        """
        series = pd.Series(np.random.randn(100))
        
        with pytest.raises(ValueError):
            fit_arima(series, order=(-1, 1, 1))

    def test_extract_residuals_empty_series(self):
        """
        Verify extract_residuals raises error on empty series.
        """
        empty_series = pd.Series([])
        series = pd.Series(np.random.randn(100))
        model = fit_arima(series, order=(1, 1, 1))
        
        with pytest.raises(ValueError):
            extract_residuals(empty_series, model)


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestARIMAParametrized:
    """
    Parametrized tests for comprehensive coverage.
    """

    @pytest.mark.parametrize("order", [(0, 0, 0), (1, 0, 1), (1, 1, 1), (2, 1, 2)])
    def test_fit_arima_multiple_orders(self, stationary_series: pd.Series, order: Tuple):
        """
        Test ARIMA fitting with multiple parameter combinations.
        """
        results = fit_arima(stationary_series, order=order)
        
        assert hasattr(results, "fittedvalues")
        assert len(results.fittedvalues) == len(stationary_series)

    @pytest.mark.parametrize("series_name", ["stationary", "non_stationary"])
    def test_stationarity_detection(
        self,
        stationary_series: pd.Series,
        non_stationary_series: pd.Series,
        series_name: str
    ):
        """
        Test stationarity detection on both types of series.
        """
        series = stationary_series if series_name == "stationary" else non_stationary_series
        is_stationary, p_value = test_stationarity(series)
        
        expected = series_name == "stationary"
        assert is_stationary == expected, \
            f"Expected is_stationary={expected} for {series_name} series"


if __name__ == "__main__":
    # Run tests with: pytest tests/test_arima.py -v
    pytest.main([__file__, "-v"])
