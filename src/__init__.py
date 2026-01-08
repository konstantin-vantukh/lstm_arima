"""
Hybrid LSTM-ARIMA Forecasting System - Core Modules

This package contains the core components for hybrid time series forecasting
combining ARIMA for linear components and LSTM for non-linear patterns.

Modules:
    - preprocessing: Data loading and preparation
    - arima_engine: ARIMA-based linear forecasting
    - lstm_engine: LSTM-based non-linear forecasting
    - hybrid_combiner: Hybrid prediction combination
    - evaluation: Metrics and validation
    - output_manager: Results export and reporting
"""

from src.preprocessing import load_data, impute_missing, calculate_returns, reshape_for_lstm
from src.arima_engine import (
    test_stationarity,
    find_optimal_params,
    fit_arima,
    extract_residuals,
)
from src.lstm_engine import (
    build_lstm_model,
    create_rolling_windows,
    train_lstm,
    predict_residuals,
)
from src.hybrid_combiner import combine_predictions
from src.evaluation import calculate_rmse, calculate_mae, walk_forward_validation
from src.output_manager import export_results, report_progress

__version__ = "1.0.0"
__author__ = "Hybrid Forecasting Team"
__all__ = [
    "load_data",
    "impute_missing",
    "calculate_returns",
    "reshape_for_lstm",
    "test_stationarity",
    "find_optimal_params",
    "fit_arima",
    "extract_residuals",
    "build_lstm_model",
    "create_rolling_windows",
    "train_lstm",
    "predict_residuals",
    "combine_predictions",
    "calculate_rmse",
    "calculate_mae",
    "walk_forward_validation",
    "export_results",
    "report_progress",
]
