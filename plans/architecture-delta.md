# LSTM Engine Updates - Architecture Delta

This document outlines the proposed changes to the LSTM engine, specifically impacting the `create_rolling_windows()` procedure and the LSTM network architecture within `./src/lstm_engine.py`. The ARIMA part of the system remains untouched.

## 1. `create_rolling_windows()` Procedure

The `create_rolling_windows()` procedure must be updated to split input data into windows with the following characteristics:
*   **Window Length:** 350 ticks.
*   **Stride:** 175 ticks, resulting in a 50% overlap between windows.

These values (window length and stride) must be configurable via the system's configuration file, with 350 and 175 serving as default values.

## 2. LSTM Network Architecture

The LSTM network must be reconfigured to comprise the following layers:

*   **First LSTM Layer:**
    *   **Nodes:** 20
    *   **`return_sequences`:** `True`

*   **Second (Hidden) LSTM Layer:**
    *   **Nodes:** 10
    *   **`return_sequences`:** `False`

*   **Dropout Layer:**
    *   **Rate:** 0.2
    *   The dropout rate must be configurable via the system's configuration file, with 0.2 being the default value.

*   **Dense Layer:**
    *   **Units:** The number of units in this layer must be dynamically set to the value provided by the `--horizon` key, which is passed as a command-line argument.

## 3. `forecaster.py` Updates

The `forecaster.py` file must be modified to replace recursive forecasting with the direct usage of the output from the Dense layer. The Dense layer's output will have a number of units equal to the forecasting horizon, eliminating the need for recursive predictions.
