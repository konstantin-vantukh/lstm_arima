# Hybrid LSTM-ARIMA Forecasting System

A sophisticated time series forecasting system that combines traditional statistical modeling (ARIMA) with deep learning (LSTM) for accurate cryptocurrency price prediction.

## Project Overview

This system decomposes time series data into linear and non-linear components:

```
x_t = L_t + N_t + ε_t
```

Where:
- **L_t** = Linear component (captured by ARIMA - AutoRegressive Integrated Moving Average)
- **N_t** = Non-linear component (captured by LSTM - Long Short-Term Memory neural networks)
- **ε_t** = Random error term

### Key Features

- **Hybrid Architecture**: Combines ARIMA's statistical rigor with LSTM's deep learning capabilities
- **Automatic Parameter Selection**: Auto-ARIMA for optimal (p,d,q) parameters minimizing AIC
- **Walk-Forward Validation**: Strict temporal train/test separation to prevent data leakage
- **Multi-format Support**: Load data from CSV or JSON formats
- **GPU Acceleration**: Optional OpenCL support for faster LSTM training
- **Comprehensive Metrics**: RMSE and MAE evaluation with detailed reporting
- **Configuration-Driven**: YAML-based model parameter configuration

## Installation

### Requirements

- Python 3.11 or higher
- Cross-platform support (Windows, Linux, macOS)

### Setup

1. **Clone the repository**:
```bash
git clone <repository_url>
cd LSTM_ARIMA_V2
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -m pytest tests/ -v
```

## Dependencies

Core dependencies managed in `requirements.txt`:

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | >= 2.0 | Data manipulation and time series handling |
| numpy | >= 1.24 | Numerical computing and matrix operations |
| statsmodels | >= 0.14 | ARIMA modeling and ADF stationarity tests |
| tensorflow | >= 2.15 | LSTM deep learning backend |
| keras | >= 3.0 | High-level LSTM API |
| scikit-learn | >= 1.3 | Metrics computation and preprocessing |
| PyYAML | >= 6.0 | Configuration file parsing |
| pyopencl | >= 2023.1 | GPU acceleration (optional) |

## Usage

### Basic CLI Command Format

```bash
python forecaster.py --input <input_file> --ticker <ticker> --horizon <periods> [--output <output_file>] [--config <config_file>]
```

### Usage Examples

**Example 1: Basic forecast with default configuration**
```bash
python forecaster.py --input data/crypto_sample.csv --ticker BTC --horizon 10
```

**Example 2: Custom output file**
```bash
python forecaster.py --input data/btc_prices.csv --ticker BTC --horizon 30 --output results/forecast.csv
```

**Example 3: With custom configuration**
```bash
python forecaster.py --input data/eth_prices.json --ticker ETH --horizon 20 --config config/custom_params.yml
```

**Example 4: Ethereum forecast**
```bash
python forecaster.py --input data/crypto_sample.csv --ticker ETH --horizon 15 --output results/eth_forecast.json
```

### CLI Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--input` | string | Yes | Path to input file (CSV or JSON) |
| `--ticker` | string | Yes | Asset ticker symbol (e.g., BTC, ETH, DOGE) |
| `--horizon` | int | Yes | Number of periods to forecast ahead |
| `--output` | string | No | Output file path (default: stdout) |
| `--config` | string | No | Path to YAML configuration file |

## Directory Structure

```
LSTM_ARIMA_V2/
├── forecaster.py                  # Main CLI entry point
├── README.md                       # Project documentation (this file)
├── requirements.txt                # Python dependencies
├── config/
│   └── model_params.yml           # Model configuration file
├── src/
│   ├── __init__.py                # Package initialization
│   ├── preprocessing.py           # Data loading and preparation
│   ├── arima_engine.py            # ARIMA-based linear forecasting
│   ├── lstm_engine.py             # LSTM-based non-linear forecasting
│   ├── hybrid_combiner.py         # Hybrid prediction combination logic
│   ├── evaluation.py              # Metrics calculation and validation
│   └── output_manager.py          # Results export and progress reporting
├── tests/
│   ├── test_arima.py              # Unit tests for ARIMA engine
│   ├── test_lstm.py               # Unit tests for LSTM engine
│   └── test_hybrid_integration.py # Integration tests
├── data/
│   └── sample/                    # Sample data files for testing
├── output/                        # Default output directory for results
└── plans/
    └── architecture.md            # Detailed system architecture document
```

## Configuration Guide

The system behavior is controlled via `config/model_params.yml`:

### ARIMA Configuration

```yaml
arima:
  seasonal: false                 # Enable/disable seasonal ARIMA (SARIMA)
  max_p: 5                        # Maximum autoregressive order
  max_d: 2                        # Maximum differencing order
  max_q: 5                        # Maximum moving average order
  information_criterion: aic      # Criterion for parameter selection
```

### LSTM Configuration

```yaml
lstm:
  hidden_layers: 1                # Number of LSTM layers
  nodes: 10                       # Neurons per layer
  batch_size: 64                  # Training batch size
  epochs: 100                     # Maximum training epochs
  dropout_rate: 0.4               # Dropout regularization for reducing overfitting
  l2_regularization: 0.01         # L2 penalty for weight regularization
  window_size: 60                 # Rolling window size for time steps
  optimizer: adam                 # Optimization algorithm
  early_stopping_patience: 10     # Patience before early stopping
```

### Validation Configuration

```yaml
validation:
  method: walk_forward            # Validation strategy (walk_forward)
  test_size: 0.2                  # Test set proportion
```

### Hardware Configuration

```yaml
hardware:
  use_opencl: true                # Enable OpenCL GPU acceleration
  gpu_memory_fraction: 0.8        # GPU memory usage limit
```

## System Architecture Components

### 1. Data Preprocessor (`src/preprocessing.py`)
- Loads data from CSV/JSON files
- Handles missing values using forward fill
- Calculates simple returns: R_t = (P_t - P_{t-1}) / P_{t-1}
- Reshapes data into 3D tensors for LSTM

### 2. ARIMA Engine (`src/arima_engine.py`)
- Tests stationarity using Augmented Dickey-Fuller (ADF) test
- Performs automatic parameter selection (Auto-ARIMA)
- Minimizes information criterion (AIC)
- Extracts residuals for LSTM processing

### 3. LSTM Engine (`src/lstm_engine.py`)
- Constructs deep learning model with gating mechanisms
- Creates rolling windows for sequential learning
- Implements early stopping to prevent overfitting
- Predicts residual patterns

### 4. Hybrid Combiner (`src/hybrid_combiner.py`)
- Combines ARIMA linear forecast with LSTM residual forecast
- Formula: ŷ_t = L̂_t + N̂_t

### 5. Evaluation Module (`src/evaluation.py`)
- Calculates RMSE (Root Mean Squared Error)
- Calculates MAE (Mean Absolute Error)
- Implements walk-forward cross-validation

### 6. Output Manager (`src/output_manager.py`)
- Exports results to CSV/JSON formats
- Reports progress to STDOUT during training
- Formats metrics for human-readable output

## Testing

Run the complete test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_arima.py -v

# Run specific test with coverage
pytest tests/test_hybrid_integration.py -v --cov=src
```

### Test Coverage

| Test File | Purpose | Coverage |
|-----------|---------|----------|
| `test_arima.py` | Unit tests for ARIMA engine | ADF test, parameter selection, fitting |
| `test_lstm.py` | Unit tests for LSTM engine | Model construction, tensor reshaping, training |
| `test_hybrid_integration.py` | End-to-end integration tests | Complete forecasting pipeline |

## Mathematical Foundations

### Simple Returns

```
R_t = (P_t - P_{t-1}) / P_{t-1}
```

Normalizes price changes for time series analysis.

### Augmented Dickey-Fuller (ADF) Test

Tests for stationarity:
- **H₀**: Series has a unit root (non-stationary)
- **H₁**: Series is stationary
- **Decision**: Reject H₀ if p-value < 0.05

### ARIMA Model

```
(1 - Σφᵢ·Lⁱ)(1 - L)ᵈ·Xₜ = (1 + Σθⱼ·Lʲ)·εₜ
```

Where:
- φᵢ = AR coefficients
- θⱼ = MA coefficients
- L = Lag operator
- d = Differencing order

### LSTM Gate Equations

**Forget gate**: f_t = σ(W_f·[h_{t-1}, x_t] + b_f)

**Input gate**: i_t = σ(W_i·[h_{t-1}, x_t] + b_i)

**Candidate**: C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)

**Cell state**: C_t = f_t·C_{t-1} + i_t·C̃_t

**Output gate**: o_t = σ(W_o·[h_{t-1}, x_t] + b_o)

**Hidden state**: h_t = o_t·tanh(C_t)

## Performance Metrics

### RMSE (Root Mean Squared Error)
```
RMSE = √(mean(residuals²))
```
Measures average magnitude of prediction errors. More sensitive to large errors.

### MAE (Mean Absolute Error)
```
MAE = mean(|residuals|)
```
Measures average absolute prediction error. More robust to outliers.

### Walk-Forward Validation

Trains on historical windows and validates on subsequent unseen data:

1. Train on: t=0 to t=n
2. Test on: t=n+1
3. Slide window forward and repeat
4. Aggregate metrics across all iterations

This prevents temporal data leakage and provides realistic performance estimates.

## Data Format Specifications

### Input Data (CSV)

```csv
timestamp,open,high,low,close,volume
2024-01-01,40000,41000,39500,40500,1000000
2024-01-02,40500,41500,40000,41000,1100000
2024-01-03,41000,42000,40500,41500,950000
```

### Input Data (JSON)

```json
{
  "data": [
    {"timestamp": "2024-01-01", "open": 40000, "high": 41000, "low": 39500, "close": 40500, "volume": 1000000},
    {"timestamp": "2024-01-02", "open": 40500, "high": 41500, "low": 40000, "close": 41000, "volume": 1100000}
  ]
}
```

### Output Format (CSV)

```csv
period,actual,arima_prediction,lstm_residual,hybrid_prediction,rmse,mae
0,40500,40480,20,40500,150.25,120.50
1,41000,40950,50,41000,145.75,118.30
```

## Limitations and Constraints

- **ARIMA Differencing**: Limited to d ≤ 2 for model stability
- **LSTM Activation**: Designed to handle residuals in range [-2, 2]
- **Data Requirements**: Minimum 100+ observations for reliable ARIMA training
- **GPU Acceleration**: Optional; CPU-only mode available for all operations

## Error Handling

The system implements graceful error handling:

| Error Type | Recovery Strategy |
|------------|-------------------|
| Missing input file | Exit with clear error message and file path |
| Non-convergent ARIMA | Log warning and apply default parameters |
| Invalid ticker | Provide list of valid options |
| GPU unavailable | Automatic fallback to CPU processing |
| Insufficient data | Exit with minimum data requirement message |

## Performance Characteristics

| Operation | CPU | GPU (OpenCL) |
|-----------|-----|--------------|
| ARIMA fitting | Default | N/A (CPU only) |
| LSTM training | Baseline | 5-10x acceleration |
| Matrix operations | NumPy | OpenCL parallelization |

## Troubleshooting

**Q: ARIMA model fails to converge**
- A: Reduce max_p/max_q values or check data quality

**Q: LSTM training is slow**
- A: Enable GPU acceleration or reduce hidden_layers/nodes

**Q: Results show large MAE despite low RMSE**
- A: Check for outliers; they may be affecting RMSE disproportionately

**Q: Forecast accuracy is poor**
- A: Verify sufficient training data, check configuration parameters, ensure data is properly normalized

## Contributing

For contributions, please:
1. Create feature branch
2. Add tests for new functionality
3. Run full test suite before submitting
4. Include documentation updates

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- Review `plans/architecture.md` for detailed technical specifications
- Check test files in `tests/` for usage examples
- Examine sample data in `data/sample/` for format specifications

## Version History

**v1.0.0** (2026-01-07)
- Initial release with hybrid LSTM-ARIMA architecture
- Complete CLI interface
- Walk-forward validation
- Comprehensive testing suite

---

*Last Updated: 2026-01-07*

*For detailed technical architecture, see [plans/architecture.md](plans/architecture.md)*
