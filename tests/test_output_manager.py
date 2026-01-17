"""
Unit tests for Output Manager Module

Tests dual-space output functions:
- export_to_csv: CSV export with returns and price space
- export_to_json: JSON export with metadata
- export_to_stdout: Human-readable console printing
- log_progress: Structured logging
- validate_output_path: Path validation and directory creation
"""

import pytest
import numpy as np
import json
import csv
from pathlib import Path
import tempfile
import shutil
import logging
from io import StringIO

# Import the output manager module
from src.output_manager import (
    export_to_csv,
    export_to_json,
    export_to_stdout,
    log_progress,
    validate_output_path,
    export_results,
    format_results_summary
)


class TestValidateOutputPath:
    """Test validate_output_path function"""
    
    def test_validate_output_path_creates_directory(self):
        """Test that validate_output_path creates parent directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "subdir" / "nested" / "forecast.csv"
            result = validate_output_path(str(test_path))
            
            assert result is True
            assert test_path.parent.exists()
    
    def test_validate_output_path_existing_directory(self):
        """Test with existing directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_output_path(str(Path(tmpdir) / "forecast.csv"))
            assert result is True
    
    def test_validate_output_path_invalid_path(self):
        """Test with invalid path returns False"""
        # Using a path that cannot be created (e.g., on restricted filesystem)
        result = validate_output_path("\0\0\0\0/forecast.csv")
        assert result is False


class TestExportToCsv:
    """Test export_to_csv function"""
    
    def test_export_to_csv_basic(self):
        """Test basic CSV export with dual-space data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_forecast.csv"
            
            # Sample data
            preds_ret = np.array([0.0125, 0.0089, -0.0034])
            preds_price = np.array([50125.50, 50570.25, 50296.18])
            arima = np.array([0.0100, 0.0080, -0.0045])
            lstm = np.array([0.0025, -0.0009, 0.0011])
            
            export_to_csv(str(output_file), preds_ret, preds_price, arima, lstm)
            
            # Verify file exists
            assert output_file.exists()
            
            # Verify content
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 3
            assert rows[0]['timestamp'] == 't1'
            assert float(rows[0]['prediction_returns']) == 0.0125
            assert float(rows[0]['prediction_price']) == 50125.50
    
    def test_export_to_csv_with_metrics(self):
        """Test CSV export with metrics included"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_metrics.csv"
            
            preds_ret = [0.01, 0.02, 0.03]
            preds_price = [100.0, 102.0, 105.0]
            arima = [0.01, 0.02, 0.03]
            lstm = [0.0, 0.0, 0.0]
            
            metrics_ret = {"rmse": 0.005, "mae": 0.003}
            metrics_price = {"rmse": 5.0, "mae": 3.0}
            
            export_to_csv(str(output_file), preds_ret, preds_price, arima, lstm,
                         metrics_ret, metrics_price)
            
            # Read and verify metrics are included
            with open(output_file, 'r') as f:
                content = f.read()
            
            assert "Metrics Summary" in content
            assert "rmse: 0.005" in content or "rmse" in content
    
    def test_export_to_csv_mismatched_lengths(self):
        """Test that mismatched array lengths raise ValueError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_mismatch.csv"
            
            preds_ret = [0.01, 0.02]
            preds_price = [100.0, 102.0, 105.0]  # Different length
            arima = [0.01, 0.02]
            lstm = [0.0, 0.0]
            
            with pytest.raises(ValueError):
                export_to_csv(str(output_file), preds_ret, preds_price, arima, lstm)


class TestExportToJson:
    """Test export_to_json function"""
    
    def test_export_to_json_basic(self):
        """Test basic JSON export"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_forecast.json"
            
            preds_ret = np.array([0.0125, 0.0089])
            preds_price = np.array([50125.50, 50570.25])
            arima = np.array([0.0100, 0.0080])
            lstm = np.array([0.0025, -0.0009])
            
            export_to_json(str(output_file), "BTC", 2, preds_ret, preds_price, arima, lstm)
            
            # Verify file exists
            assert output_file.exists()
            
            # Verify content
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            assert data["ticker"] == "BTC"
            assert data["horizon"] == 2
            assert len(data["predictions_returns"]) == 2
            assert len(data["predictions_price"]) == 2
            assert "timestamp" in data
    
    def test_export_to_json_with_metrics_and_params(self):
        """Test JSON export with metrics and model parameters"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_full.json"
            
            preds_ret = [0.01, 0.02]
            preds_price = [100.0, 102.0]
            arima = [0.01, 0.02]
            lstm = [0.0, 0.0]
            
            metrics_ret = {"rmse": 0.005, "mae": 0.003}
            metrics_price = {"rmse": 5.0, "mae": 3.0}
            model_params = {"arima_order": [1, 1, 1], "last_price": 100.0}
            
            export_to_json(str(output_file), "ETH", 2, preds_ret, preds_price, 
                          arima, lstm, metrics_ret, metrics_price, model_params)
            
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            assert "metrics_returns" in data
            assert "metrics_price" in data
            assert "model_params" in data
            assert data["model_params"]["arima_order"] == [1, 1, 1]
    
    def test_export_to_json_invalid_inputs(self):
        """Test JSON export with invalid inputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_invalid.json"
            
            # Invalid ticker (empty string)
            with pytest.raises(ValueError):
                export_to_json(str(output_file), "", 2, [0.01], [100.0], [0.01], [0.0])
            
            # Invalid horizon (negative)
            with pytest.raises(ValueError):
                export_to_json(str(output_file), "BTC", -1, [0.01], [100.0], [0.01], [0.0])


class TestExportToStdout:
    """Test export_to_stdout function"""
    
    def test_export_to_stdout_basic(self, capsys):
        """Test stdout export displays predictions"""
        preds_ret = np.array([0.0125, 0.0089, -0.0034, 0.0045, 0.0067])
        preds_price = np.array([50125.50, 50570.25, 50296.18, 50518.42, 50850.99])
        
        export_to_stdout(preds_ret, preds_price)
        
        captured = capsys.readouterr()
        
        # Verify output contains expected sections
        assert "HYBRID LSTM-ARIMA FORECAST" in captured.out
        assert "First" in captured.out
        assert "FORECAST SUMMARY" in captured.out
        assert "Total Predictions" in captured.out
    
    def test_export_to_stdout_with_metrics(self, capsys):
        """Test stdout export with metrics"""
        preds_ret = [0.01, 0.02]
        preds_price = [100.0, 102.0]
        metrics_ret = {"rmse": 0.005, "mae": 0.003}
        metrics_price = {"rmse": 5.0, "mae": 3.0}
        
        export_to_stdout(preds_ret, preds_price, metrics_ret, metrics_price)
        
        captured = capsys.readouterr()
        
        assert "PERFORMANCE METRICS" in captured.out
        assert "Returns Space" in captured.out
        assert "Price Space" in captured.out


class TestLogProgress:
    """Test log_progress function"""
    
    def test_log_progress_info(self, caplog):
        """Test log_progress with INFO level"""
        with caplog.at_level(logging.INFO):
            log_progress("Test message", level='INFO')
        
        assert "Test message" in caplog.text
    
    def test_log_progress_warning(self, caplog):
        """Test log_progress with WARNING level"""
        with caplog.at_level(logging.WARNING):
            log_progress("Warning message", level='WARNING')
        
        assert "Warning message" in caplog.text
    
    def test_log_progress_invalid_level(self):
        """Test log_progress with invalid level"""
        with pytest.raises(ValueError):
            log_progress("Test", level='DEBUG')
    
    def test_log_progress_invalid_message_type(self):
        """Test log_progress with non-string message"""
        with pytest.raises(TypeError):
            log_progress(123, level='INFO')


class TestExportResults:
    """Test legacy export_results function"""
    
    def test_export_results_csv(self):
        """Test legacy export_results with CSV format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "legacy_export.csv"
            
            results = {
                'predictions': np.array([100.5, 101.2, 102.1]),
                'arima_component': np.array([100.0, 101.0, 102.0]),
                'lstm_component': np.array([0.5, 0.2, 0.1]),
                'metrics': {'rmse': 0.35, 'mae': 0.27}
            }
            
            path = export_results(results, str(output_file), format='csv')
            
            assert Path(path).exists()
            assert Path(path).suffix == '.csv'
    
    def test_export_results_json(self):
        """Test legacy export_results with JSON format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "legacy_export.json"
            
            results = {
                'predictions': [100.5, 101.2, 102.1],
                'arima_component': [100.0, 101.0, 102.0],
                'lstm_component': [0.5, 0.2, 0.1]
            }
            
            path = export_results(results, str(output_file), format='json')
            
            assert Path(path).exists()
            assert Path(path).suffix == '.json'


class TestFormatResultsSummary:
    """Test format_results_summary function"""
    
    def test_format_results_summary_basic(self):
        """Test results summary formatting"""
        results = {
            'predictions': np.array([100.5, 101.2, 102.1]),
            'ticker': 'BTC',
            'horizon': 3,
            'metrics': {'rmse': 0.35, 'mae': 0.27}
        }
        
        summary = format_results_summary(results)
        
        assert "FORECAST RESULTS SUMMARY" in summary
        assert "BTC" in summary
        assert "RMSE" in summary
    
    def test_format_results_summary_missing_predictions(self):
        """Test with missing predictions key"""
        results = {'ticker': 'BTC'}
        
        with pytest.raises(ValueError):
            format_results_summary(results)


def test_integration_dual_space_export():
    """Integration test: export full dual-space forecast"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample dual-space data
        horizon = 5
        preds_ret = np.array([0.01, 0.015, -0.005, 0.02, 0.012])
        preds_price = np.array([50100, 50175, 50162, 50324, 50436])
        arima_comp = np.array([0.008, 0.012, -0.003, 0.018, 0.010])
        lstm_comp = np.array([0.002, 0.003, -0.002, 0.002, 0.002])
        
        metrics_ret = {"rmse": 0.0045, "mae": 0.0032}
        metrics_price = {"rmse": 225.50, "mae": 160.25}
        model_params = {"arima_order": [1, 1, 1], "last_price": 50000}
        
        # Test CSV export
        csv_file = str(Path(tmpdir) / "forecast.csv")
        export_to_csv(csv_file, preds_ret, preds_price, arima_comp, lstm_comp,
                     metrics_ret, metrics_price)
        assert Path(csv_file).exists()
        
        # Test JSON export
        json_file = str(Path(tmpdir) / "forecast.json")
        export_to_json(json_file, "BTC", horizon, preds_ret, preds_price,
                      arima_comp, lstm_comp, metrics_ret, metrics_price, model_params)
        assert Path(json_file).exists()
        
        # Verify JSON structure
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        assert data["ticker"] == "BTC"
        assert data["horizon"] == horizon
        assert len(data["predictions_returns"]) == horizon
        assert len(data["predictions_price"]) == horizon
        assert "arima_component" in data
        assert "lstm_component" in data


if __name__ == "__main__":
    # Run tests with: pytest tests/test_output_manager.py -v
    pytest.main([__file__, "-v"])
