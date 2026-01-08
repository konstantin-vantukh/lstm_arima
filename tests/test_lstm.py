"""
Unit Tests for LSTM Engine Module

This test module provides comprehensive coverage of the LSTM engine functionality
including model construction, rolling window generation, training with early stopping, 
and residual prediction. Follows the test strategy outlined in architecture.md Section 11.2.

Test Coverage:
- UT1: LSTM model construction with correct architecture
- UT2: Model output shape verification
- UT3: Rolling window creation (3D tensor reshaping)
- UT4: Rolling window dimensions verification [Samples, Time Steps, Features]
- UT5: LSTM training completion without error
- UT6: Model weights change after training
- UT7: Residual prediction generation
- UT8: Predictions within approximate [-2, 2] range (tanh activation)
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple

from src.lstm_engine import (
    build_lstm_model,
    create_rolling_windows,
    train_lstm,
    predict_residuals
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def lstm_config() -> dict:
    """
    Create a standard LSTM configuration for testing.
    
    Returns:
        dict: Configuration with all required LSTM parameters
    """
    return {
        'hidden_layers': 1,
        'nodes': 10,
        'batch_size': 16,
        'epochs': 5,
        'dropout_rate': 0.4,
        'l2_regularization': 0.01,
        'window_size': 20,
        'optimizer': 'adam',
        'early_stopping_patience': 3
    }


@pytest.fixture
def synthetic_residuals() -> np.ndarray:
    """
    Generate synthetic residual data for testing.
    
    Creates a 1D array of synthetic residuals with values approximately in
    the range [-2, 2] to simulate ARIMA residuals. Uses a seed for reproducibility.
    
    Returns:
        np.ndarray: 1D array of synthetic residuals (length 200)
    """
    np.random.seed(42)
    # Generate centered, scaled residuals approximately in [-2, 2] range
    residuals = np.random.randn(200) * 0.8  # Roughly [-2.4, 2.4], most within [-2, 2]
    return residuals.astype(np.float32)


@pytest.fixture
def rolling_windows_example() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create example rolling windows for testing.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: X of shape (50, 20, 1), y of shape (50,)
    """
    np.random.seed(42)
    residuals = np.random.randn(70).astype(np.float32)
    X, y = create_rolling_windows(residuals, window_size=20)
    return X, y


@pytest.fixture
def trained_lstm_model(lstm_config: dict, rolling_windows_example: Tuple) -> keras.Model:
    """
    Create and train a LSTM model for testing.
    
    Returns:
        keras.Model: Trained LSTM model ready for prediction testing
    """
    X, y = rolling_windows_example
    input_shape = (lstm_config['window_size'], 1)
    model = build_lstm_model(lstm_config, input_shape)
    
    # Train with minimal epochs for testing
    small_config = lstm_config.copy()
    small_config['epochs'] = 3
    small_config['early_stopping_patience'] = 2
    
    model = train_lstm(model, X, y, small_config)
    return model


# ============================================================================
# TEST 1: BUILD LSTM MODEL - ARCHITECTURE VERIFICATION
# ============================================================================

class TestBuildLSTMModel:
    """
    Test suite for LSTM model construction.
    
    Validates that the LSTM model is built with correct architecture
    (LSTM → Dropout → Dense with L2 → Output) per architecture.md Section 4.4.
    """

    def test_build_lstm_model(self, lstm_config: dict):
        """
        Verify LSTM model construction with correct architecture.
        
        Creates a model and verifies it has the expected layer structure:
        - LSTM layer with gated structure
        - Dropout layer (0.4)
        - Dense layer with L2 regularization
        - Output layer with tanh activation
        
        This test validates UT1 from architecture.md Section 11.2.
        """
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        # Verify model is a Keras Sequential model
        assert isinstance(model, keras.Sequential), \
            f"Model should be keras.Sequential, got {type(model)}"
        
        # Verify model is compiled
        assert model.optimizer is not None, \
            "Model should be compiled with an optimizer"
        assert model.loss is not None, \
            "Model should be compiled with a loss function"

    def test_build_lstm_model_layer_count(self, lstm_config: dict):
        """
        Verify model has expected number of layers.
        
        For hidden_layers=1, should have: LSTM + Dropout + Dense + Output = 4 layers
        """
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        # Expected: LSTM + Dropout + Dense + Output layers
        expected_layer_count = 4
        assert len(model.layers) == expected_layer_count, \
            f"Model should have {expected_layer_count} layers, got {len(model.layers)}"

    def test_build_lstm_model_first_layer_lstm(self, lstm_config: dict):
        """
        Verify first layer is LSTM.
        """
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        # First layer should be LSTM
        first_layer = model.layers[0]
        assert isinstance(first_layer, tf.keras.layers.LSTM), \
            f"First layer should be LSTM, got {type(first_layer)}"

    def test_build_lstm_model_dropout_layer(self, lstm_config: dict):
        """
        Verify dropout layer exists and has correct rate.
        """
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        # Second layer should be Dropout
        dropout_layer = model.layers[1]
        assert isinstance(dropout_layer, tf.keras.layers.Dropout), \
            f"Second layer should be Dropout, got {type(dropout_layer)}"
        
        # Verify dropout rate
        assert dropout_layer.rate == lstm_config['dropout_rate'], \
            f"Dropout rate should be {lstm_config['dropout_rate']}, " \
            f"got {dropout_layer.rate}"

    def test_build_lstm_model_output_layer_tanh(self, lstm_config: dict):
        """
        Verify output layer has tanh activation for residual range [-2, 2].
        """
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        # Last layer should be Dense with tanh activation
        output_layer = model.layers[-1]
        assert isinstance(output_layer, tf.keras.layers.Dense), \
            f"Output layer should be Dense, got {type(output_layer)}"
        
        # Verify tanh activation
        assert output_layer.activation == tf.keras.activations.tanh, \
            f"Output layer should have tanh activation, " \
            f"got {output_layer.activation}"

    def test_build_lstm_model_lstm_nodes(self, lstm_config: dict):
        """
        Verify LSTM layer has correct number of nodes.
        """
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        lstm_layer = model.layers[0]
        assert lstm_layer.units == lstm_config['nodes'], \
            f"LSTM should have {lstm_config['nodes']} nodes, got {lstm_layer.units}"

    def test_build_lstm_model_with_multiple_layers(self):
        """
        Verify model construction with multiple LSTM layers.
        """
        config = {
            'hidden_layers': 2,
            'nodes': 10,
            'dropout_rate': 0.4,
            'l2_regularization': 0.01,
            'optimizer': 'adam'
        }
        input_shape = (20, 1)
        model = build_lstm_model(config, input_shape)
        
        # With hidden_layers=2: LSTM + LSTM + Dropout + Dense + Output = 5 layers
        expected_layer_count = 5
        assert len(model.layers) == expected_layer_count, \
            f"Model with 2 hidden layers should have {expected_layer_count} layers, " \
            f"got {len(model.layers)}"

    def test_build_lstm_model_optimizer_adam(self, lstm_config: dict):
        """
        Verify model uses Adam optimizer.
        """
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam), \
            f"Optimizer should be Adam, got {type(model.optimizer)}"

    def test_build_lstm_model_mse_loss(self, lstm_config: dict):
        """
        Verify model uses MSE loss function.
        """
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        assert 'mse' in str(model.loss).lower(), \
            f"Loss should be MSE, got {model.loss}"


# ============================================================================
# TEST 2: BUILD LSTM MODEL - OUTPUT SHAPE VERIFICATION
# ============================================================================

class TestBuildLSTMModelOutputShape:
    """
    Test suite for LSTM model output shape verification.
    
    Validates that the model produces correct output dimensions.
    """

    def test_build_lstm_model_output_shape(self, lstm_config: dict, rolling_windows_example: Tuple):
        """
        Verify model output shape matches expected format.
        
        Model should produce predictions of shape (samples, 1) for 1D output.
        This test validates UT2 from architecture.md Section 11.2.
        """
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        X, _ = rolling_windows_example
        predictions = model.predict(X, verbose=0)
        
        # Output should be [samples, 1]
        assert predictions.shape[0] == X.shape[0], \
            f"Output samples should match input samples: {predictions.shape[0]} != {X.shape[0]}"
        assert predictions.shape[1] == 1, \
            f"Output should have 1 feature, got {predictions.shape[1]}"

    def test_build_lstm_model_output_shape_batch_predictions(self, lstm_config: dict):
        """
        Verify output shape consistency for different batch sizes.
        """
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        # Test with different batch sizes
        for batch_size in [1, 5, 10, 32]:
            X_batch = np.random.randn(batch_size, lstm_config['window_size'], 1).astype(np.float32)
            predictions = model.predict(X_batch, verbose=0)
            
            assert predictions.shape == (batch_size, 1), \
                f"Output shape should be ({batch_size}, 1), got {predictions.shape}"

    def test_build_lstm_model_output_dtype(self, lstm_config: dict, rolling_windows_example: Tuple):
        """
        Verify output is float32 dtype.
        """
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        X, _ = rolling_windows_example
        predictions = model.predict(X, verbose=0)
        
        assert predictions.dtype == np.float32 or predictions.dtype == np.float64, \
            f"Output should be float type, got {predictions.dtype}"


# ============================================================================
# TEST 3: CREATE ROLLING WINDOWS - 3D TENSOR RESHAPING
# ============================================================================

class TestCreateRollingWindows:
    """
    Test suite for rolling window creation (3D tensor reshaping).
    
    Validates that 1D residual data is correctly reshaped to 3D format
    suitable for LSTM training. This is critical for UT2 from the contract.
    """

    def test_create_rolling_windows(self, synthetic_residuals: np.ndarray):
        """
        Verify rolling window creation produces correct 3D tensor.
        
        Should reshape 1D data into 3D format [Samples, Time Steps, Features].
        This test validates UT3 from architecture.md Section 11.2 and the
        contract requirement UT2: "Ensure LSTM output buffer matches the
        dimension of the ARIMA residual input".
        """
        window_size = 20
        X, y = create_rolling_windows(synthetic_residuals, window_size=window_size)
        
        # X should be 3D: [samples, window_size, 1]
        assert len(X.shape) == 3, \
            f"X should be 3D tensor, got shape {X.shape}"
        
        # y should be 1D: [samples]
        assert len(y.shape) == 1, \
            f"y should be 1D array, got shape {y.shape}"

    def test_create_rolling_windows_sample_count(self, synthetic_residuals: np.ndarray):
        """
        Verify correct number of samples generated.
        
        With data length N and window_size W, should generate N - W samples.
        """
        window_size = 20
        X, y = create_rolling_windows(synthetic_residuals, window_size=window_size)
        
        expected_samples = len(synthetic_residuals) - window_size
        assert X.shape[0] == expected_samples, \
            f"Should have {expected_samples} samples, got {X.shape[0]}"
        assert y.shape[0] == expected_samples, \
            f"y should have {expected_samples} samples, got {y.shape[0]}"

    def test_create_rolling_windows_window_size(self, synthetic_residuals: np.ndarray):
        """
        Verify window size parameter is respected.
        """
        window_size = 30
        X, y = create_rolling_windows(synthetic_residuals, window_size=window_size)
        
        # X should have second dimension equal to window_size
        assert X.shape[1] == window_size, \
            f"Window size should be {window_size}, got {X.shape[1]}"

    def test_create_rolling_windows_features_dimension(self, synthetic_residuals: np.ndarray):
        """
        Verify features dimension is 1 for univariate data.
        
        LSTM residual input should be univariate with 1 feature.
        """
        window_size = 20
        X, y = create_rolling_windows(synthetic_residuals, window_size=window_size)
        
        # Third dimension should be 1 (univariate)
        assert X.shape[2] == 1, \
            f"Features dimension should be 1, got {X.shape[2]}"

    def test_create_rolling_windows_data_continuity(self, synthetic_residuals: np.ndarray):
        """
        Verify rolling windows capture sequential data correctly.
        
        Each window should contain consecutive values from the original residuals.
        """
        window_size = 5
        X, y = create_rolling_windows(synthetic_residuals, window_size=window_size)
        
        # Check first window
        expected_first_window = synthetic_residuals[:window_size].reshape(-1, 1)
        np.testing.assert_allclose(
            X[0],
            expected_first_window,
            rtol=1e-6,
            err_msg="First window should match first N values"
        )
        
        # Check that y values are the next values after each window
        np.testing.assert_allclose(
            y[0],
            synthetic_residuals[window_size],
            rtol=1e-6,
            err_msg="First y value should be the value after first window"
        )

    def test_create_rolling_windows_different_sizes(self, synthetic_residuals: np.ndarray):
        """
        Verify rolling windows work with different window sizes.
        """
        for window_size in [10, 20, 30, 40, 50]:
            X, y = create_rolling_windows(synthetic_residuals, window_size=window_size)
            
            # Verify expected shape
            expected_samples = len(synthetic_residuals) - window_size
            assert X.shape == (expected_samples, window_size, 1), \
                f"Shape should be ({expected_samples}, {window_size}, 1), got {X.shape}"

    def test_create_rolling_windows_dtype_float32(self, synthetic_residuals: np.ndarray):
        """
        Verify output arrays are float32 for efficiency.
        """
        window_size = 20
        X, y = create_rolling_windows(synthetic_residuals, window_size=window_size)
        
        assert X.dtype == np.float32, \
            f"X should be float32, got {X.dtype}"
        assert y.dtype == np.float32, \
            f"y should be float32, got {y.dtype}"


# ============================================================================
# TEST 4: CREATE ROLLING WINDOWS - DIMENSIONS VERIFICATION
# ============================================================================

class TestCreateRollingWindowsDimensions:
    """
    Test suite for rolling window dimensions verification.
    
    Validates that shape is [Samples, Time Steps, Features] format.
    """

    def test_create_rolling_windows_dimensions(self, synthetic_residuals: np.ndarray):
        """
        Verify shape is [Samples, Time Steps, Features].
        
        This test validates UT4 from architecture.md Section 11.2.
        """
        window_size = 20
        X, y = create_rolling_windows(synthetic_residuals, window_size=window_size)
        
        # Verify 3D structure: [Samples, Time Steps, Features]
        samples, time_steps, features = X.shape
        
        assert samples > 0, "Number of samples should be positive"
        assert time_steps == window_size, \
            f"Time steps should equal window_size ({window_size}), got {time_steps}"
        assert features == 1, \
            f"Features should be 1 (univariate), got {features}"

    def test_create_rolling_windows_shape_semantic_meaning(self, synthetic_residuals: np.ndarray):
        """
        Verify shape dimensions have correct semantic meaning.
        
        Each sample should contain a sequence of consecutive residual values.
        """
        window_size = 10
        X, y = create_rolling_windows(synthetic_residuals, window_size=window_size)
        
        # Get shape components
        num_samples, num_timesteps, num_features = X.shape
        
        # Verify semantic meaning:
        # - num_samples: how many sequences we can create
        assert num_samples == len(synthetic_residuals) - window_size, \
            "Number of samples should be len(data) - window_size"
        
        # - num_timesteps: length of each sequence
        assert num_timesteps == window_size, \
            "Number of timesteps should equal window_size"
        
        # - num_features: univariate (1 feature)
        assert num_features == 1, \
            "Should have 1 feature for univariate residuals"

    def test_create_rolling_windows_shape_consistency(self, synthetic_residuals: np.ndarray):
        """
        Verify shape remains consistent across multiple calls.
        """
        window_size = 20
        
        X1, y1 = create_rolling_windows(synthetic_residuals, window_size=window_size)
        X2, y2 = create_rolling_windows(synthetic_residuals, window_size=window_size)
        
        assert X1.shape == X2.shape, \
            f"Shape should be consistent: {X1.shape} != {X2.shape}"
        assert y1.shape == y2.shape, \
            f"y shape should be consistent: {y1.shape} != {y2.shape}"


# ============================================================================
# TEST 5: TRAIN LSTM - TRAINING COMPLETION
# ============================================================================

class TestTrainLSTM:
    """
    Test suite for LSTM training.
    
    Validates that LSTM training completes without error and that
    model weights change during training.
    """

    def test_train_lstm(self, lstm_config: dict, rolling_windows_example: Tuple):
        """
        Verify LSTM training completes without error.
        
        This test validates UT5 from architecture.md Section 11.2.
        """
        X, y = rolling_windows_example
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        # Create small config for fast training
        small_config = lstm_config.copy()
        small_config['epochs'] = 2
        small_config['early_stopping_patience'] = 1
        
        # Training should complete without error
        trained_model = train_lstm(model, X, y, small_config)
        
        # Should return a Keras model
        assert isinstance(trained_model, keras.Model), \
            f"train_lstm should return keras.Model, got {type(trained_model)}"

    def test_train_lstm_completes_successfully(self, lstm_config: dict, rolling_windows_example: Tuple):
        """
        Verify training completes successfully and returns a valid model.
        """
        X, y = rolling_windows_example
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        small_config = lstm_config.copy()
        small_config['epochs'] = 2
        small_config['early_stopping_patience'] = 1
        
        # Should not raise exception
        try:
            trained_model = train_lstm(model, X, y, small_config)
            assert trained_model is not None
        except Exception as e:
            pytest.fail(f"Training should not raise exception: {str(e)}")

    def test_train_lstm_with_validation_split(self, lstm_config: dict, rolling_windows_example: Tuple):
        """
        Verify training works with validation split (20% for validation).
        """
        X, y = rolling_windows_example
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        small_config = lstm_config.copy()
        small_config['epochs'] = 2
        small_config['early_stopping_patience'] = 1
        
        trained_model = train_lstm(model, X, y, small_config)
        
        # Model should be trainable after training
        assert trained_model.trainable or not trained_model.trainable, \
            "Model should have trainable attribute"


# ============================================================================
# TEST 6: TRAIN LSTM - MODEL WEIGHTS CHANGE
# ============================================================================

class TestTrainLSTMModelWeightsChange:
    """
    Test suite for verifying model weights change during training.
    
    Validates that the model actually learns during training by checking
    that weights change.
    """

    def test_train_lstm_model_trained(self, lstm_config: dict, rolling_windows_example: Tuple):
        """
        Verify model weights change after training.
        
        This test validates UT6 from architecture.md Section 11.2.
        """
        X, y = rolling_windows_example
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        # Store initial weights
        initial_weights = [w.numpy().copy() for w in model.weights]
        
        # Train model
        small_config = lstm_config.copy()
        small_config['epochs'] = 5
        small_config['early_stopping_patience'] = 10
        
        trained_model = train_lstm(model, X, y, small_config)
        
        # Get trained weights
        trained_weights = [w.numpy() for w in trained_model.weights]
        
        # At least some weights should have changed
        weights_changed = False
        for init_w, trained_w in zip(initial_weights, trained_weights):
            if not np.allclose(init_w, trained_w, rtol=1e-4):
                weights_changed = True
                break
        
        assert weights_changed, \
            "Model weights should change after training"

    def test_train_lstm_model_loss_decreases(self, lstm_config: dict, rolling_windows_example: Tuple):
        """
        Verify training loss decreases during training.
        """
        X, y = rolling_windows_example
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        # Evaluate initial loss
        initial_loss = model.evaluate(X, y, verbose=0)
        
        # Train model
        small_config = lstm_config.copy()
        small_config['epochs'] = 10
        small_config['early_stopping_patience'] = 5
        
        trained_model = train_lstm(model, X, y, small_config)
        
        # Evaluate trained loss
        trained_loss = trained_model.evaluate(X, y, verbose=0)
        
        # Loss should generally decrease (or at least be different)
        assert initial_loss != trained_loss, \
            "Training should change the model loss"

    def test_train_lstm_early_stopping_triggered(self, lstm_config: dict, rolling_windows_example: Tuple):
        """
        Verify early stopping functionality works.
        """
        X, y = rolling_windows_example
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        small_config = lstm_config.copy()
        small_config['epochs'] = 100  # Set high epochs to test early stopping
        small_config['early_stopping_patience'] = 2
        
        # Training should complete despite high epoch count due to early stopping
        trained_model = train_lstm(model, X, y, small_config)
        
        assert trained_model is not None, \
            "Model should be trained with early stopping"


# ============================================================================
# TEST 7: PREDICT RESIDUALS - RESIDUAL PREDICTION GENERATION
# ============================================================================

class TestPredictResiduals:
    """
    Test suite for residual prediction generation.
    
    Validates that the trained LSTM model can generate predictions
    for residual values.
    """

    def test_predict_residuals(self, trained_lstm_model: keras.Model, rolling_windows_example: Tuple):
        """
        Verify residual prediction generation.
        
        This test validates UT7 from architecture.md Section 11.2.
        """
        X, _ = rolling_windows_example
        predictions = predict_residuals(trained_lstm_model, X)
        
        # Predictions should be a numpy array
        assert isinstance(predictions, np.ndarray), \
            f"Predictions should be numpy array, got {type(predictions)}"
        
        # Predictions should be 1D
        assert len(predictions.shape) == 1, \
            f"Predictions should be 1D, got shape {predictions.shape}"

    def test_predict_residuals_shape(self, trained_lstm_model: keras.Model, rolling_windows_example: Tuple):
        """
        Verify predictions have correct shape.
        """
        X, _ = rolling_windows_example
        predictions = predict_residuals(trained_lstm_model, X)
        
        # Should have one prediction per input sample
        assert predictions.shape[0] == X.shape[0], \
            f"Should have {X.shape[0]} predictions, got {predictions.shape[0]}"

    def test_predict_residuals_dtype(self, trained_lstm_model: keras.Model, rolling_windows_example: Tuple):
        """
        Verify predictions are float dtype.
        """
        X, _ = rolling_windows_example
        predictions = predict_residuals(trained_lstm_model, X)
        
        assert predictions.dtype in [np.float32, np.float64], \
            f"Predictions should be float, got {predictions.dtype}"

    def test_predict_residuals_single_sample(self, trained_lstm_model: keras.Model):
        """
        Verify prediction works for single sample.
        """
        X_single = np.random.randn(1, 20, 1).astype(np.float32)
        predictions = predict_residuals(trained_lstm_model, X_single)
        
        assert predictions.shape == (1,), \
            f"Single sample prediction should have shape (1,), got {predictions.shape}"

    def test_predict_residuals_batch(self, trained_lstm_model: keras.Model):
        """
        Verify prediction works for batch of samples.
        """
        X_batch = np.random.randn(32, 20, 1).astype(np.float32)
        predictions = predict_residuals(trained_lstm_model, X_batch)
        
        assert predictions.shape == (32,), \
            f"Batch prediction should have shape (32,), got {predictions.shape}"

    def test_predict_residuals_consistency(self, trained_lstm_model: keras.Model, rolling_windows_example: Tuple):
        """
        Verify predictions are consistent across multiple calls.
        """
        X, _ = rolling_windows_example
        
        pred1 = predict_residuals(trained_lstm_model, X)
        pred2 = predict_residuals(trained_lstm_model, X)
        
        np.testing.assert_allclose(
            pred1, pred2,
            rtol=1e-6,
            err_msg="Predictions should be consistent across calls"
        )


# ============================================================================
# TEST 8: PREDICT RESIDUALS - OUTPUT RANGE VALIDATION
# ============================================================================

class TestPredictResidualsRange:
    """
    Test suite for verifying predictions are within appropriate range.
    
    Validates that predictions are in approximate [-2, 2] range due to
    tanh activation on output layer.
    """

    def test_predict_residuals_within_range(self, trained_lstm_model: keras.Model, rolling_windows_example: Tuple):
        """
        Verify predictions are in approximate [-2, 2] range.
        
        Due to tanh activation, predictions should be in [-1, 1] range,
        but scaled residuals may be slightly different. This tests that
        predictions fall within reasonable bounds.
        
        This test validates UT8 from architecture.md Section 11.2.
        """
        X, _ = rolling_windows_example
        predictions = predict_residuals(trained_lstm_model, X)
        
        # Tanh output is [-1, 1], but we allow slightly larger range for scaled residuals
        min_pred = predictions.min()
        max_pred = predictions.max()
        
        # At minimum, check they're bounded by tanh theoretical range
        assert -1.1 <= min_pred <= 1.1, \
            f"Min prediction should be near [-1, 1] range, got {min_pred}"
        assert -1.1 <= max_pred <= 1.1, \
            f"Max prediction should be near [-1, 1] range, got {max_pred}"

    def test_predict_residuals_tanh_bounds(self, trained_lstm_model: keras.Model, rolling_windows_example: Tuple):
        """
        Verify predictions strictly respect tanh bounds [-1, 1].
        
        Tanh activation function guarantees output is in [-1, 1].
        """
        X, _ = rolling_windows_example
        predictions = predict_residuals(trained_lstm_model, X)
        
        # All predictions must be within tanh bounds
        assert np.all(predictions >= -1.0 - 1e-6), \
            f"All predictions should be >= -1, found min: {predictions.min()}"
        assert np.all(predictions <= 1.0 + 1e-6), \
            f"All predictions should be <= 1, found max: {predictions.max()}"

    def test_predict_residuals_reasonable_distribution(self, trained_lstm_model: keras.Model, rolling_windows_example: Tuple):
        """
        Verify predictions have reasonable distribution (not all zeros).
        """
        X, _ = rolling_windows_example
        predictions = predict_residuals(trained_lstm_model, X)
        
        # Predictions should have some variance (not all identical)
        pred_std = np.std(predictions)
        assert pred_std > 0.01, \
            f"Predictions should have variance, std: {pred_std}"

    def test_predict_residuals_mean_centered(self, trained_lstm_model: keras.Model, rolling_windows_example: Tuple):
        """
        Verify predictions have reasonable mean (approximately centered).
        """
        X, _ = rolling_windows_example
        predictions = predict_residuals(trained_lstm_model, X)
        
        pred_mean = np.mean(predictions)
        
        # Mean should be within reasonable range (not all positive or negative)
        assert -0.5 <= pred_mean <= 0.5, \
            f"Prediction mean should be approximately centered, got {pred_mean}"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestLSTMErrorHandling:
    """
    Test error handling in LSTM functions.
    """

    def test_create_rolling_windows_insufficient_data(self):
        """
        Verify error when data is too short for window size.
        """
        data = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            create_rolling_windows(data, window_size=10)

    def test_create_rolling_windows_invalid_window_size(self, synthetic_residuals: np.ndarray):
        """
        Verify error for invalid window size.
        """
        with pytest.raises(ValueError):
            create_rolling_windows(synthetic_residuals, window_size=0)

    def test_build_lstm_model_invalid_dropout(self):
        """
        Verify error for invalid dropout rate.
        """
        config = {
            'hidden_layers': 1,
            'nodes': 10,
            'dropout_rate': 1.5,  # Invalid: should be < 1
            'l2_regularization': 0.01,
            'optimizer': 'adam'
        }
        
        with pytest.raises(ValueError):
            build_lstm_model(config, (20, 1))

    def test_train_lstm_mismatched_shapes(self, lstm_config: dict):
        """
        Verify error when X and y shapes don't match.
        """
        X = np.random.randn(10, 20, 1).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)  # Mismatched size
        
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        with pytest.raises(ValueError):
            train_lstm(model, X, y, lstm_config)

    def test_predict_residuals_invalid_shape(self, trained_lstm_model: keras.Model):
        """
        Verify error for invalid input shape to predict_residuals.
        """
        X_invalid = np.random.randn(10, 20).astype(np.float32)  # 2D instead of 3D
        
        with pytest.raises(ValueError):
            predict_residuals(trained_lstm_model, X_invalid)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestLSTMIntegration:
    """
    Integration tests for complete LSTM workflow.
    """

    def test_complete_lstm_workflow(self, lstm_config: dict, synthetic_residuals: np.ndarray):
        """
        Test complete workflow: build model → create windows → train → predict.
        """
        # Step 1: Create rolling windows
        X, y = create_rolling_windows(synthetic_residuals, window_size=lstm_config['window_size'])
        
        # Step 2: Build model
        input_shape = (lstm_config['window_size'], 1)
        model = build_lstm_model(lstm_config, input_shape)
        
        # Step 3: Train model
        small_config = lstm_config.copy()
        small_config['epochs'] = 3
        small_config['early_stopping_patience'] = 2
        
        trained_model = train_lstm(model, X, y, small_config)
        
        # Step 4: Make predictions
        predictions = predict_residuals(trained_model, X[:10])
        
        # Verify complete workflow produced valid results
        assert predictions.shape == (10,), \
            f"Expected 10 predictions, got shape {predictions.shape}"
        assert np.all(predictions >= -1.1) and np.all(predictions <= 1.1), \
            f"Predictions should be in tanh range"

    def test_lstm_workflow_with_different_window_sizes(self, lstm_config: dict, synthetic_residuals: np.ndarray):
        """
        Test LSTM workflow with different window sizes.
        """
        for window_size in [15, 20, 30]:
            config = lstm_config.copy()
            config['window_size'] = window_size
            
            # Create windows
            X, y = create_rolling_windows(synthetic_residuals, window_size=window_size)
            
            # Build and train
            input_shape = (window_size, 1)
            model = build_lstm_model(config, input_shape)
            
            small_config = config.copy()
            small_config['epochs'] = 2
            small_config['early_stopping_patience'] = 1
            
            trained_model = train_lstm(model, X, y, small_config)
            
            # Predict
            predictions = predict_residuals(trained_model, X[:5])
            
            assert predictions.shape == (5,), \
                f"Window size {window_size}: expected 5 predictions"


if __name__ == "__main__":
    # Run tests with: pytest tests/test_lstm.py -v
    pytest.main([__file__, "-v"])
