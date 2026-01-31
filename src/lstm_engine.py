"""
LSTM Engine Module for Hybrid LSTM-ARIMA Forecasting System

This module provides LSTM (Long Short-Term Memory) functionality for modeling
non-linear residual patterns in time series forecasting. It handles model construction,
rolling window generation, training with early stopping, and residual prediction.

Functions:
    - build_lstm_model: Construct LSTM architecture with gated structure
    - create_rolling_windows: Generate training sequences (X, y arrays)
    - train_lstm: Train LSTM model with early stopping
    - predict_residuals: Generate non-linear predictions on residuals

Error Handling & Logging: Section 10, Error Handling Strategy
"""

import logging
import numpy as np
import traceback
import time
from typing import Tuple, Dict, Any, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.exceptions import ConfigurationError, DataValidationError
from src.logger_config import get_logger, log_exception


# Configure module logging
logger = get_logger(__name__)


def build_lstm_model(config: Dict[str, Any], input_shape: Tuple[int, int]) -> keras.Model:
    """
    Construct LSTM neural network architecture for residual modeling.

    Builds a sequential LSTM model with the following layers:
    - First LSTM layer: 20 nodes, return_sequences=True
    - Hidden LSTM layer: 10 nodes, return_sequences=False
    - Dropout layer for regularization (rate configurable in config)
     - Output Dense layer: units equal to the forecasting horizon, with tanh activation (handles residual range -2 to 2)

    Args:
        config (Dict[str, Any]): Model configuration dictionary containing:
            - lstm.dropout_rate (float): Dropout rate (default: 0.2)
            - lstm.l2_regularization (float): L2 regularization coefficient (default: 0.01)
            - horizon (int): The number of units for the Dense layer (default: 1)
        input_shape (Tuple[int, int]): Shape of input data (window_size, features)
            - First element: number of time steps in window
            - Second element: number of features (univariate = 1)

    Returns:
        keras.Model: Compiled Keras Sequential model ready for training

    Raises:
        ConfigurationError: If config parameters are invalid or missing
        ValueError: If input_shape is malformed
        Exception: For any Keras model building failures

    Examples:
        >>> config = {
        ...     'lstm': {'dropout_rate': 0.2, 'l2_regularization': 0.01},
        ...     'horizon': 1
        ... }
        >>> input_shape = (350, 1)  # Example: 350 time steps, 1 feature
        >>> model = build_lstm_model(config, input_shape)
        >>> print(model.summary())
    """
    # Extract configuration parameters with defaults
    dropout_rate = config.get('lstm', {}).get('dropout_rate', config.get('dropout_rate', 0.2))
    l2_reg = config.get('lstm', {}).get('l2_regularization', config.get('l2_regularization', 0.01))
    # The 'horizon' value is typically passed as a command-line argument,
    # and we assume it's available in the config dictionary.
    # Default to 1 for single-step prediction if not found.
    horizon = config.get('horizon', 1) 

    # Validate configuration
    if not (0 <= dropout_rate < 1):
        error_msg = f"dropout_rate must be in [0, 1), got {dropout_rate}"
        logger.error(error_msg)
        raise ConfigurationError(
            error_message=error_msg,
            parameter_name="dropout_rate",
            invalid_value=dropout_rate,
            allowed_range="[0, 1)"
        )

    if l2_reg < 0:
        error_msg = f"l2_regularization must be non-negative, got {l2_reg}"
        logger.error(error_msg)
        raise ConfigurationError(
            error_message=error_msg,
            parameter_name="l2_regularization",
            invalid_value=l2_reg,
            allowed_range="[0, ∞)"
        )
    
    if horizon <= 0:
        error_msg = f"horizon must be positive, got {horizon}"
        logger.error(error_msg)
        raise ConfigurationError(
            error_message=error_msg,
            parameter_name="horizon",
            invalid_value=horizon,
            allowed_range="(0, ∞)"
        )

    # Validate input shape
    if len(input_shape) != 2:
        error_msg = f"input_shape must be 2D (window_size, features), got {len(input_shape)}D"
        logger.error(error_msg)
        raise ValueError(error_msg)

    window_size, features = input_shape
    if window_size <= 0 or features <= 0:
        error_msg = f"input_shape dimensions must be positive, got {input_shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.info(
            f"Building LSTM model: dropout_rate={dropout_rate}, l2_regularization={l2_reg}, "
            f"horizon={horizon}, input_shape={input_shape}"
        )

        model = keras.Sequential()

        # First LSTM layer: 20 nodes, return_sequences=True
        model.add(
            LSTM(
                units=20,
                activation='tanh',
                input_shape=input_shape,
                return_sequences=True
            )
        )
        logger.debug(f"Added First LSTM layer: units=20, return_sequences=True")

        # Hidden LSTM layer: 10 nodes, return_sequences=False
        model.add(
            LSTM(
                units=10,
                activation='tanh',
                return_sequences=False
            )
        )
        logger.debug(f"Added Hidden LSTM layer: units=10, return_sequences=False")

        # Dropout layer for regularization
        model.add(Dropout(rate=dropout_rate))
        logger.debug(f"Added Dropout layer: rate={dropout_rate}")

        # Dense layer with L2 regularization, now serving as the output layer
        model.add(
            Dense(
                units=horizon,
                activation='tanh',
                kernel_regularizer=l2(l2_reg)
            )
        )
        logger.debug(f"Added Output Dense layer: units={horizon}, activation=tanh, l2_regularization={l2_reg}")

        # Compile model with Adam optimizer
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        logger.info("Model compiled successfully with Adam optimizer, MSE loss, MAE metric")

        return model

    except ConfigurationError:
        raise
    except Exception as e:
        error_msg = f"Failed to build LSTM model: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise


def create_rolling_windows(
    data: np.ndarray, config: Optional[Dict[str, Any]] = None, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training sequences using rolling window approach with configurable window size and stride.

    Converts a 1D array into training pairs (X, y) suitable for LSTM training.
    X represents input sequences of length window_size, and y represents the
    next `horizon` values in the sequence to be predicted.

    Args:
        data (np.ndarray): 1D array of residual or sequential data
        config (Dict[str, Any]): Configuration dictionary containing:
            - lstm.window_size (int): Size of rolling window (number of time steps)
            - lstm.stride (int): Stride for rolling window (number of steps to advance)
            - horizon (int): The number of future steps to predict per window

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - X: 3D array of shape [samples, window_size, 1] - input sequences
            - y: 2D array of shape [samples, horizon] - target values (next `horizon` time steps)

    Raises:
        ValueError: If data is invalid or window_size/stride are inappropriate
        TypeError: If data is not a numpy array
        ConfigurationError: If config parameters are invalid or missing

    Examples:
        >>> config = {'lstm': {'window_size': 3, 'stride': 1}, 'horizon': 2}
        >>> residuals = np.array([-0.1, 0.05, -0.02, 0.08, 0.01, -0.03, 0.04, 0.02])
        >>> X, y = create_rolling_windows(residuals, config)
        >>> X.shape
        (4, 3, 1)
        >>> y.shape
        (4, 2)
    """
    # Extract configuration parameters with defaults
    if config is None:
        config = {}
    
    window_size = config.get('lstm', {}).get('window_size', config.get('window_size', kwargs.get('window_size', 350)))
    stride = config.get('lstm', {}).get('stride', config.get('stride', kwargs.get('stride', 175)))
    horizon = config.get('horizon', kwargs.get('horizon', 1))  # Get horizon (target prediction length)

    # Validate configuration for horizon
    if not isinstance(horizon, int) or horizon <= 0:
        error_msg = f"horizon must be a positive integer, got {horizon}"
        logger.error(error_msg)
        raise ConfigurationError(
            error_message=error_msg,
            parameter_name="horizon",
            invalid_value=horizon,
            allowed_range="(0, ∞)"
        )

    # Validate input type
    if not isinstance(data, np.ndarray):
        error_msg = "Data must be a numpy array"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Validate data dimensionality
    if len(data.shape) != 1:
        error_msg = f"Data must be 1D array, got shape {data.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate window size
    if not isinstance(window_size, int) or window_size <= 0:
        error_msg = f"window_size must be a positive integer, got {window_size}"
        logger.error(error_msg)
        raise ConfigurationError(
            error_message=error_msg,
            parameter_name="lstm.window_size",
            invalid_value=window_size,
            allowed_range="(0, ∞)"
        )

    # Validate stride
    if not isinstance(stride, int) or stride <= 0:
        error_msg = f"stride must be a positive integer, got {stride}"
        logger.error(error_msg)
        raise ConfigurationError(
            error_message=error_msg,
            parameter_name="lstm.stride",
            invalid_value=stride,
            allowed_range="(0, ∞)"
        )

    if window_size < stride:
        error_msg = f"window_size ({window_size}) cannot be less than stride ({stride})"
        logger.error(error_msg)
        raise ConfigurationError(
            error_message=error_msg,
            parameter_name="lstm.stride",
            invalid_value=stride,
            allowed_range=f"(0, {window_size}]"
        )

    # Check sufficient data for both window and horizon
    if len(data) < window_size + horizon:
        error_msg = (
            f"Data length ({len(data)}) must be at least window_size ({window_size}) "
            f"+ horizon ({horizon}) for rolling window creation."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.info(
            f"Creating rolling windows: data_length={len(data)}, window_size={window_size}, "
            f"stride={stride}, horizon={horizon}"
        )

        X = []
        y = []

        # Generate rolling windows
        for i in range(0, len(data) - window_size - horizon + 1, stride):
            # Extract window of consecutive values and reshape to [window_size, 1]
            window = data[i : i + window_size].reshape(-1, 1)
            X.append(window)

            # Target is now a sequence of 'horizon' values
            target_sequence = data[i + window_size : i + window_size + horizon]
            y.append(target_sequence)


        # If X is empty, it means no windows could be formed due to data length or window/stride sizes
        if not X or not y or len(X) != len(y):
            error_msg = (
                f"No valid windows or targets could be created with data length {len(data)}, "
                f"window_size {window_size}, stride {stride}, and horizon {horizon}. "
                "Ensure data length is sufficient and configuration allows window formation."
            )
            logger.error(error_msg)
            raise DataValidationError(error_message=error_msg)

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32) # y is now 2D: [samples, horizon]

        num_samples = len(X)
        logger.info(
            f"Rolling windows created: X shape={X.shape}, y shape={y.shape}, "
            f"num_samples={num_samples}"
        )

        return X, y

    except Exception as e:
        error_msg = f"Failed to create rolling windows: {str(e)}"
        logger.error(error_msg)
        raise


def train_lstm(
    model: keras.Model, X: np.ndarray, y: np.ndarray, config: Dict[str, Any]
) -> keras.Model:
    """
    Train LSTM model with early stopping to prevent overfitting.

    Trains the LSTM model on the provided training data with early stopping
    based on validation loss. Outputs training progress to STDOUT.

    Args:
        model (keras.Model): Compiled Keras LSTM model from build_lstm_model
        X (np.ndarray): Training input sequences of shape [samples, window_size, features]
        y (np.ndarray): Training target values of shape [samples, horizon]
        config (Dict[str, Any]): Training configuration containing:
            - batch_size (int): Training batch size (default: 64)
            - epochs (int): Maximum number of training epochs (default: 100)
            - early_stopping_patience (int): Patience for early stopping (default: 10)

    Returns:
        keras.Model: Trained Keras model with optimized weights

    Raises:
        ValueError: If data shapes are incompatible or config is invalid
        TypeError: If inputs are of incorrect types
        Exception: For any Keras training failures

    Examples:
        >>> config = {'batch_size': 64, 'epochs': 100, 'early_stopping_patience': 10}
        >>> X_train = np.random.randn(100, 60, 1)  # 100 samples, 60 time steps, 1 feature
        >>> y_train = np.random.randn(100)
        >>> trained_model = train_lstm(model, X_train, y_train, config)

    Note:
        - Early stopping returns the epoch with lowest validation loss
        - Default batch_size: 64
        - Default max epochs: 100
        - Default early_stopping patience: 10 epochs with no improvement
        - Training progress (epochs, loss) printed to STDOUT
        - Validation split: 20% of data used for validation
    """
    # Extract configuration parameters with defaults
    batch_size = config.get('batch_size', 64)
    epochs = config.get('epochs', 100)
    early_stopping_patience = config.get('early_stopping_patience', 10)

    # Validate configuration
    if not isinstance(batch_size, int) or batch_size <= 0:
        error_msg = f"batch_size must be positive integer, got {batch_size}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(epochs, int) or epochs <= 0:
        error_msg = f"epochs must be positive integer, got {epochs}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(early_stopping_patience, int) or early_stopping_patience <= 0:
        error_msg = f"early_stopping_patience must be positive integer, got {early_stopping_patience}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate input data types
    if not isinstance(X, np.ndarray):
        error_msg = f"X must be numpy array, got {type(X)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if not isinstance(y, np.ndarray):
        error_msg = f"y must be numpy array, got {type(y)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Validate data shapes
    if len(X.shape) != 3:
        error_msg = f"X must be 3D array [samples, window_size, features], got shape {X.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(y.shape) != 2:
        error_msg = f"y must be 2D array [samples, horizon], got shape {y.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate y's second dimension (horizon)
    horizon = config.get('horizon', 1)
    if y.shape[1] != horizon:
        error_msg = (
            f"Second dimension of y ({y.shape[1]}) must match horizon ({horizon})."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if X.shape[0] != y.shape[0]:
        error_msg = (
            f"X and y must have same number of samples. "
            f"X: {X.shape[0]}, y: {y.shape[0]}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Verify batch_size is not larger than data
    if batch_size > X.shape[0]:
        logger.warning(
            f"batch_size ({batch_size}) exceeds data size ({X.shape[0]}). "
            f"Adjusting batch_size to data size."
        )
        batch_size = X.shape[0]

    try:
        logger.info(
            f"Starting LSTM training: batch_size={batch_size}, epochs={epochs}, "
            f"early_stopping_patience={early_stopping_patience}, "
            f"training_samples={X.shape[0]}"
        )

        # Set up early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )

        # Train model with progress output to STDOUT
        print("\n" + "="*80)
        print("LSTM Training Progress")
        print("="*80)

        history = model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,  # 20% for validation, 80% for training
            callbacks=[early_stop],
            verbose=1  # Print progress to STDOUT
        )

        print("="*80)
        print(f"Training completed. Best epoch: {len(history.history['loss']) - early_stopping_patience}")
        print("="*80 + "\n")

        # Log final training statistics
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_mae = history.history['mae'][-1] if 'mae' in history.history else 0.0

        logger.info(
            f"LSTM training completed. Final loss={final_loss:.6f}, "
            f"final val_loss={final_val_loss:.6f}, final mae={final_mae:.6f}, "
            f"epochs_trained={len(history.history['loss'])}"
        )

        return model

    except Exception as e:
        error_msg = f"LSTM training failed: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise


def predict_residuals(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Generate non-linear residual predictions using trained LSTM model.

    Uses the trained LSTM model to predict residual values for input sequences.
    Predictions are made on sequences of shape [samples, window_size, features].

    Args:
        model (keras.Model): Trained Keras LSTM model from train_lstm
        X (np.ndarray): Input sequences of shape [samples, window_size, features]

    Returns:
        np.ndarray: Predicted residuals of shape [samples]
            - Values typically in range approximately [-2, 2] due to tanh activation
            - One prediction per input sequence (specifically the first step of the multi-step forecast)

    Raises:
        ValueError: If X shape is incompatible with model
        TypeError: If inputs are of incorrect types
        Exception: For any Keras prediction failures

    Examples:
        >>> X_test = np.random.randn(10, 60, 1)  # 10 samples, 60 time steps, 1 feature
        >>> predictions = predict_residuals(model, X_test)
        >>> predictions.shape
        (10,)
        >>> print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

    Note:
        - Predictions are flattened to 1D array
        - Output activation is tanh, so values are in [-1, 1] range
        - Used to capture non-linear patterns in ARIMA residuals
        - Together with ARIMA predictions, forms hybrid forecast: y = ARIMA + LSTM
    """
    # Validate input type
    if not isinstance(X, np.ndarray):
        error_msg = f"X must be numpy array, got {type(X)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Validate input shape
    if len(X.shape) != 3:
        error_msg = f"X must be 3D array [samples, window_size, features], got shape {X.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Check that data is not empty
    if X.shape[0] == 0:
        error_msg = "X contains no samples"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.info(f"Generating predictions for {X.shape[0]} samples")

        # Generate predictions
        predictions = model.predict(X, verbose=0)


        logger.info(
            f"Predictions generated: shape={predictions.shape}, "
            f"mean={predictions.mean():.6f}, std={predictions.std():.6f}, "
            f"min={predictions.min():.6f}, max={predictions.max():.6f}"
        )

        # If horizon is 1, squeeze to 1D array for compatibility
        if predictions.shape[1] == 1:
            return predictions.squeeze(axis=1)
        return predictions

    except Exception as e:
        error_msg = f"Residual prediction failed: {str(e)}"
        logger.error(error_msg)
        log_exception(logger, e)
        raise
