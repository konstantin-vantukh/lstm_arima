"""
Simple integration test for Output Manager Module
Tests dual-space export functionality with sample data
"""

import numpy as np
import tempfile
from pathlib import Path
import json
import csv
import sys

# Add src to path
sys.path.insert(0, 'src')

from output_manager import (
    export_to_csv,
    export_to_json,
    export_to_stdout,
    log_progress,
    validate_output_path
)

def test_validate_output_path():
    """Test path validation"""
    print("\n" + "="*60)
    print("TEST 1: validate_output_path()")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = str(Path(tmpdir) / "subdir" / "forecast.csv")
        result = validate_output_path(test_path)
        print(f"[OK] Path validation result: {result}")
        print(f"[OK] Directory created: {Path(test_path).parent.exists()}")
        assert result == True
        assert Path(test_path).parent.exists()
    
    print("[PASS] validate_output_path()")

def test_export_to_csv():
    """Test CSV export"""
    print("\n" + "="*60)
    print("TEST 2: export_to_csv()")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = str(Path(tmpdir) / "forecast.csv")
        
        # Create sample dual-space data
        preds_ret = np.array([0.0125, 0.0089, -0.0034])
        preds_price = np.array([50125.50, 50570.25, 50296.18])
        arima = np.array([0.0100, 0.0080, -0.0045])
        lstm = np.array([0.0025, -0.0009, 0.0011])
        
        metrics_ret = {"rmse": 0.0045, "mae": 0.0032}
        metrics_price = {"rmse": 225.50, "mae": 160.25}
        
        export_to_csv(output_file, preds_ret, preds_price, arima, lstm,
                     metrics_ret, metrics_price)
        
        # Verify file exists
        assert Path(output_file).exists()
        print(f"[OK] CSV file created: {output_file}")
        
        # Verify content
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if row['timestamp'] and not row['timestamp'].startswith('#')]
        
        print(f"[OK] CSV has {len(rows)} data rows")
        print(f"[OK] First row: {rows[0]}")
        
        assert len(rows) >= 3
        assert rows[0]['timestamp'] == 't1'
        assert float(rows[0]['prediction_returns']) == 0.0125
        
    print("[PASS] export_to_csv()")

def test_export_to_json():
    """Test JSON export"""
    print("\n" + "="*60)
    print("TEST 3: export_to_json()")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = str(Path(tmpdir) / "forecast.json")
        
        # Create sample data
        preds_ret = np.array([0.0125, 0.0089])
        preds_price = np.array([50125.50, 50570.25])
        arima = np.array([0.0100, 0.0080])
        lstm = np.array([0.0025, -0.0009])
        
        metrics_ret = {"rmse": 0.0045, "mae": 0.0032}
        metrics_price = {"rmse": 225.50, "mae": 160.25}
        model_params = {"arima_order": [1, 1, 1], "last_price": 50000.00}
        
        export_to_json(output_file, "BTC", 2, preds_ret, preds_price, arima, lstm,
                      metrics_ret, metrics_price, model_params)
        
        # Verify file exists
        assert Path(output_file).exists()
        print(f"[OK] JSON file created: {output_file}")
        
        # Verify content
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        print(f"[OK] JSON ticker: {data['ticker']}")
        print(f"[OK] JSON horizon: {data['horizon']}")
        print(f"[OK] Prediction count (returns): {len(data['predictions_returns'])}")
        print(f"[OK] Prediction count (price): {len(data['predictions_price'])}")
        
        assert data["ticker"] == "BTC"
        assert data["horizon"] == 2
        assert len(data["predictions_returns"]) == 2
        assert len(data["predictions_price"]) == 2
        assert "metrics_returns" in data
        assert "model_params" in data
        
    print("[PASS] export_to_json()")

def test_export_to_stdout():
    """Test STDOUT export"""
    print("\n" + "="*60)
    print("TEST 4: export_to_stdout()")
    print("="*60)
    
    preds_ret = np.array([0.0125, 0.0089, -0.0034, 0.0045, 0.0067])
    preds_price = np.array([50125.50, 50570.25, 50296.18, 50518.42, 50850.99])
    metrics_ret = {"rmse": 0.0045, "mae": 0.0032}
    metrics_price = {"rmse": 225.50, "mae": 160.25}
    
    print("Calling export_to_stdout()...")
    export_to_stdout(preds_ret, preds_price, metrics_ret, metrics_price)
    
    print("[PASS] export_to_stdout()")

def test_log_progress():
    """Test log_progress"""
    print("\n" + "="*60)
    print("TEST 5: log_progress()")
    print("="*60)
    
    log_progress("Starting ARIMA model fitting", level='INFO')
    print("[OK] INFO level logging works")
    
    log_progress("GPU memory warning", level='WARNING')
    print("[OK] WARNING level logging works")
    
    print("[PASS] log_progress()")

def test_integration():
    """Integration test: Full dual-space export pipeline"""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Full Dual-Space Export")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create realistic forecast data
        horizon = 10
        np.random.seed(42)
        
        # Generate data
        base_price = 50000
        returns = np.random.normal(0.001, 0.02, horizon)
        prices = np.cumprod(1 + returns) * base_price
        
        # ARIMA and LSTM components
        arima_comp = returns * 0.7  # ARIMA captures 70%
        lstm_comp = returns * 0.3   # LSTM captures remainder
        
        # Metrics
        metrics_ret = {"rmse": 0.0045, "mae": 0.0032}
        metrics_price = {"rmse": 225.50, "mae": 160.25}
        model_params = {"arima_order": [1, 1, 1], "last_price": base_price}
        
        # Export to CSV
        csv_file = str(Path(tmpdir) / "forecast_complete.csv")
        export_to_csv(csv_file, returns, prices, arima_comp, lstm_comp,
                     metrics_ret, metrics_price)
        print(f"[OK] CSV exported: {csv_file}")
        
        # Export to JSON
        json_file = str(Path(tmpdir) / "forecast_complete.json")
        export_to_json(json_file, "BTC", horizon, returns, prices, 
                      arima_comp, lstm_comp, metrics_ret, metrics_price, model_params)
        print(f"[OK] JSON exported: {json_file}")
        
        # Verify both files exist and have content
        assert Path(csv_file).exists() and Path(csv_file).stat().st_size > 0
        assert Path(json_file).exists() and Path(json_file).stat().st_size > 0
        
        print(f"[OK] CSV file size: {Path(csv_file).stat().st_size} bytes")
        print(f"[OK] JSON file size: {Path(json_file).stat().st_size} bytes")
        
    print("[PASS] Integration Test")

if __name__ == "__main__":
    try:
        test_validate_output_path()
        test_export_to_csv()
        test_export_to_json()
        test_export_to_stdout()
        test_log_progress()
        test_integration()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED [OK]")
        print("="*60)
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
