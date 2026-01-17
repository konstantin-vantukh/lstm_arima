"""
Unit Tests for Configuration System Module (src/config_loader.py)

Tests core configuration loading, validation, and merging functionality
per Architecture Section 7 requirements.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

# Import configuration module
from src.config_loader import (
    load_config,
    validate_config,
    get_default_config,
    merge_config,
    config_to_dict,
    ConfigurationError
)


class TestGetDefaultConfig:
    """Test get_default_config() function."""
    
    def test_default_config_structure(self):
        """Test that default config has all required sections."""
        config = get_default_config()
        
        required_sections = ['arima', 'lstm', 'validation', 'hardware', 'output']
        for section in required_sections:
            assert section in config, f"Missing section: {section}"
    
    def test_arima_defaults(self):
        """Test ARIMA default parameters."""
        config = get_default_config()
        arima = config['arima']
        
        assert arima['seasonal'] == False
        assert arima['max_p'] == 5
        assert arima['max_d'] == 2
        assert arima['max_q'] == 5
        assert arima['information_criterion'] == 'aic'
    
    def test_lstm_defaults(self):
        """Test LSTM default parameters."""
        config = get_default_config()
        lstm = config['lstm']
        
        assert lstm['hidden_layers'] == 1
        assert lstm['nodes'] == 10
        assert lstm['batch_size'] == 64
        assert lstm['epochs'] == 100
        assert lstm['dropout_rate'] == 0.4
        assert lstm['l2_regularization'] == 0.01
        assert lstm['window_size'] == 60
        assert lstm['optimizer'] == 'adam'
        assert lstm['early_stopping_patience'] == 10
    
    def test_validation_defaults(self):
        """Test validation default parameters."""
        config = get_default_config()
        validation = config['validation']
        
        assert validation['method'] == 'walk_forward'
        assert validation['test_size'] == 0.2
    
    def test_hardware_defaults(self):
        """Test hardware default parameters."""
        config = get_default_config()
        hardware = config['hardware']
        
        assert hardware['use_opencl'] == True
        assert hardware['gpu_memory_fraction'] == 0.8
    
    def test_output_defaults(self):
        """Test output default parameters (dual-space)."""
        config = get_default_config()
        output = config['output']
        
        assert output['include_returns_space'] == True
        assert output['include_price_space'] == True
        assert output['metrics_in_both_spaces'] == True


class TestValidateConfig:
    """Test validate_config() function."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = get_default_config()
        assert validate_config(config) == True
    
    def test_missing_section(self):
        """Test validation fails when section is missing."""
        config = get_default_config()
        del config['arima']
        
        with pytest.raises(ConfigurationError):
            validate_config(config)
    
    def test_arima_max_p_validation(self):
        """Test ARIMA max_p validation."""
        config = get_default_config()
        
        # Test invalid (zero)
        config['arima']['max_p'] = 0
        with pytest.raises(ConfigurationError):
            validate_config(config)
        
        # Test valid (positive)
        config['arima']['max_p'] = 5
        assert validate_config(config) == True
    
    def test_lstm_nodes_range(self):
        """Test LSTM nodes must be in range 5-20."""
        config = get_default_config()
        
        # Test too low
        config['lstm']['nodes'] = 3
        with pytest.raises(ConfigurationError):
            validate_config(config)
        
        # Test too high
        config['lstm']['nodes'] = 25
        with pytest.raises(ConfigurationError):
            validate_config(config)
        
        # Test valid
        config['lstm']['nodes'] = 10
        assert validate_config(config) == True
    
    def test_dropout_rate_range(self):
        """Test dropout rate must be 0.0-1.0."""
        config = get_default_config()
        
        # Test too low
        config['lstm']['dropout_rate'] = -0.1
        with pytest.raises(ConfigurationError):
            validate_config(config)
        
        # Test too high
        config['lstm']['dropout_rate'] = 1.5
        with pytest.raises(ConfigurationError):
            validate_config(config)
        
        # Test valid
        config['lstm']['dropout_rate'] = 0.4
        assert validate_config(config) == True
    
    def test_test_size_range(self):
        """Test validation test_size must be 0.1-0.5."""
        config = get_default_config()
        
        # Test too low
        config['validation']['test_size'] = 0.05
        with pytest.raises(ConfigurationError):
            validate_config(config)
        
        # Test too high
        config['validation']['test_size'] = 0.7
        with pytest.raises(ConfigurationError):
            validate_config(config)
        
        # Test valid
        config['validation']['test_size'] = 0.2
        assert validate_config(config) == True
    
    def test_gpu_memory_range(self):
        """Test GPU memory fraction must be 0.1-1.0."""
        config = get_default_config()
        
        # Test too low
        config['hardware']['gpu_memory_fraction'] = 0.05
        with pytest.raises(ConfigurationError):
            validate_config(config)
        
        # Test too high
        config['hardware']['gpu_memory_fraction'] = 1.5
        with pytest.raises(ConfigurationError):
            validate_config(config)
        
        # Test valid
        config['hardware']['gpu_memory_fraction'] = 0.8
        assert validate_config(config) == True
    
    def test_dual_space_constraint(self):
        """Test that both returns and price spaces must be enabled."""
        config = get_default_config()
        
        # Test returns space disabled
        config['output']['include_returns_space'] = False
        with pytest.raises(ConfigurationError):
            validate_config(config)
        
        # Test price space disabled
        config['output']['include_returns_space'] = True
        config['output']['include_price_space'] = False
        with pytest.raises(ConfigurationError):
            validate_config(config)
        
        # Test both enabled (valid)
        config['output']['include_price_space'] = True
        assert validate_config(config) == True


class TestLoadConfig:
    """Test load_config() function."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()
        assert config is not None
        assert 'arima' in config
        assert 'lstm' in config
    
    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        # Use existing config file
        config = load_config('config/model_params.yml')
        assert config is not None
        assert 'arima' in config
        assert config['arima']['max_p'] == 5
    
    def test_missing_file_fallback(self):
        """Test fallback to defaults when file missing."""
        config = load_config('config/nonexistent.yml')
        assert config is not None
        # Should use defaults
        assert config['lstm']['nodes'] == 10
    
    def test_loaded_config_validated(self):
        """Test that loaded config is automatically validated."""
        config = load_config()
        # If we got here, validation passed
        assert validate_config(config) == True


class TestMergeConfig:
    """Test merge_config() function."""
    
    def test_simple_merge(self):
        """Test simple configuration merge."""
        base = get_default_config()
        overrides = {'lstm': {'nodes': 20}}
        
        merged = merge_config(base, overrides)
        
        assert merged['lstm']['nodes'] == 20
        assert merged['lstm']['batch_size'] == 64  # Preserved
    
    def test_deep_merge(self):
        """Test deep merge of nested structures."""
        base = get_default_config()
        overrides = {
            'arima': {'max_p': 10},
            'lstm': {'nodes': 15}
        }
        
        merged = merge_config(base, overrides)
        
        assert merged['arima']['max_p'] == 10
        assert merged['arima']['max_d'] == 2  # Preserved
        assert merged['lstm']['nodes'] == 15
        assert merged['lstm']['epochs'] == 100  # Preserved
    
    def test_merge_preserves_base(self):
        """Test that merge doesn't modify original base."""
        base = get_default_config()
        original_nodes = base['lstm']['nodes']
        
        overrides = {'lstm': {'nodes': 20}}
        merged = merge_config(base, overrides)
        
        # Base should be unchanged
        assert base['lstm']['nodes'] == original_nodes
        # Merged should have override
        assert merged['lstm']['nodes'] == 20
    
    def test_merge_validates_result(self):
        """Test that merged config is validated."""
        base = get_default_config()
        overrides = {'lstm': {'nodes': 100}}  # Invalid
        
        with pytest.raises(ConfigurationError):
            merge_config(base, overrides)
    
    def test_empty_overrides(self):
        """Test merge with empty overrides."""
        base = get_default_config()
        merged = merge_config(base, {})
        
        assert merged == base


class TestConfigToDict:
    """Test config_to_dict() function."""
    
    def test_flatten_config(self):
        """Test flattening of nested config."""
        config = get_default_config()
        flat = config_to_dict(config)
        
        assert isinstance(flat, dict)
        assert 'arima.max_p' in flat
        assert flat['arima.max_p'] == '5'
    
    def test_all_keys_present(self):
        """Test that all keys are flattened."""
        config = get_default_config()
        flat = config_to_dict(config)
        
        # Count nested keys
        expected_keys = 0
        for section in config.values():
            if isinstance(section, dict):
                expected_keys += len(section)
        
        assert len(flat) == expected_keys
    
    def test_boolean_conversion(self):
        """Test boolean values are converted correctly."""
        config = get_default_config()
        flat = config_to_dict(config)
        
        assert flat['output.include_returns_space'] == 'true'
        assert flat['output.include_price_space'] == 'true'
    
    def test_numeric_conversion(self):
        """Test numeric values are converted to strings."""
        config = get_default_config()
        flat = config_to_dict(config)
        
        assert flat['lstm.nodes'] == '10'
        assert flat['lstm.batch_size'] == '64'


class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    def test_full_workflow(self):
        """Test complete configuration workflow."""
        # 1. Load config
        config = load_config()
        
        # 2. Validate config
        assert validate_config(config) == True
        
        # 3. Merge overrides
        overrides = {'lstm': {'nodes': 15}}
        merged = merge_config(config, overrides)
        
        # 4. Validate merged
        assert validate_config(merged) == True
        assert merged['lstm']['nodes'] == 15
        
        # 5. Flatten for display
        flat = config_to_dict(merged)
        assert 'lstm.nodes' in flat
        assert flat['lstm.nodes'] == '15'
    
    def test_yaml_file_integration(self):
        """Test loading from actual YAML file."""
        # Load from existing config file
        config = load_config('config/model_params.yml')
        
        # Should be valid
        assert validate_config(config) == True
        
        # Should have correct structure
        assert config['arima']['max_p'] == 5
        assert config['lstm']['nodes'] == 10
        assert config['output']['include_returns_space'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
