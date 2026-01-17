"""
Configuration System Module - Centralized Model Parameter Management

This module provides a complete configuration management system for the Hybrid LSTM-ARIMA
forecasting system. It handles loading, validating, and merging configuration parameters
from YAML files and CLI overrides.

Per Architecture Section 7, this module centralizes all model parameters including:
- ARIMA configuration (operates on returns space)
- LSTM configuration (operates on returns space residuals)
- Validation configuration (walk-forward validation)
- Hardware configuration (GPU/CPU selection)
- Output configuration (dual-space processing)

Configuration can be loaded from file or overridden via CLI parameters, enabling flexible
deployment and experimentation without code changes.

Usage Examples:
    # Load default configuration
    config = load_config()
    
    # Load from specific file
    config = load_config('config/custom_params.yml')
    
    # Validate configuration
    if validate_config(config):
        print("Configuration valid")
    
    # Merge CLI overrides
    overrides = {'lstm': {'nodes': 20}}
    merged = merge_config(config, overrides)
    
    # Get defaults
    defaults = get_default_config()
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


def get_default_config() -> Dict[str, Any]:
    """
    Return hardcoded default configuration.
    
    Provides fallback configuration when YAML parsing fails or no config file is
    specified. Values match architecture Section 7 specification exactly.
    
    Returns:
        dict: Complete default configuration with all required sections:
              - arima: ARIMA parameters (returns space)
              - lstm: LSTM parameters (returns space residuals)
              - validation: Validation configuration
              - hardware: Hardware acceleration settings
              - output: Output space configuration (dual-space processing)
    
    Examples:
        >>> config = get_default_config()
        >>> config['arima']['max_p']
        5
        >>> config['lstm']['nodes']
        10
    """
    default_config = {
        # ARIMA Configuration (operates on returns space)
        'arima': {
            'seasonal': False,
            'max_p': 5,
            'max_d': 2,
            'max_q': 5,
            'information_criterion': 'aic'
        },
        # LSTM Configuration (operates on returns space residuals)
        'lstm': {
            'hidden_layers': 1,
            'nodes': 10,
            'batch_size': 64,
            'epochs': 100,
            'dropout_rate': 0.4,
            'l2_regularization': 0.01,
            'window_size': 60,
            'optimizer': 'adam',
            'early_stopping_patience': 10
        },
        # Validation Configuration
        'validation': {
            'method': 'walk_forward',
            'test_size': 0.2
        },
        # Hardware Configuration
        'hardware': {
            'use_opencl': True,
            'gpu_memory_fraction': 0.8
        },
        # Output Configuration
        'output': {
            'include_returns_space': True,
            'include_price_space': True,
            'metrics_in_both_spaces': True
        }
    }
    
    logger.debug("Default configuration created")
    return default_config


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or use defaults.
    
    Attempts to load configuration from the specified path. If no path is provided,
    uses default config/model_params.yml. If file doesn't exist or parsing fails,
    falls back to hardcoded defaults. Validates configuration after loading.
    
    Per Architecture Section 7:
    - Supports YAML format configuration files
    - YAML file should be at config/model_params.yml by default
    - If no file exists, uses get_default_config() fallback
    - Validates all required keys are present
    - Logs loaded configuration values
    
    Args:
        config_path (str, optional): Path to YAML configuration file.
                                    Default is 'config/model_params.yml'.
                                    If None, uses default path.
    
    Returns:
        dict: Complete validated configuration dictionary with all sections.
    
    Raises:
        ConfigurationError: If configuration validation fails after loading.
    
    Examples:
        >>> # Load default configuration
        >>> config = load_config()
        >>> 
        >>> # Load from custom path
        >>> config = load_config('config/custom_params.yml')
        >>> 
        >>> # Load with fallback to defaults if file doesn't exist
        >>> config = load_config('config/missing_file.yml')
        >>> config['arima']['max_p']  # Gets default value
        5
    """
    # Use default path if none provided
    if config_path is None:
        config_path = 'config/model_params.yml'
    
    config_path = Path(config_path)
    config = None
    
    # Try to load from file
    if config_path.exists():
        try:
            logger.info(f"Loading configuration from {config_path}")
            
            if not YAML_AVAILABLE:
                logger.warning(
                    "PyYAML not available, attempting fallback to JSON format"
                )
                # Try JSON fallback
                if config_path.suffix.lower() == '.json':
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    logger.info("Successfully loaded configuration from JSON")
                else:
                    logger.warning(
                        f"Cannot parse {config_path} without PyYAML. "
                        "Falling back to defaults."
                    )
            else:
                # Load YAML file
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Successfully loaded configuration from {config_path}")
        
        except yaml.YAMLError as e:
            logger.warning(
                f"YAML parsing error in {config_path}: {str(e)}. "
                "Falling back to default configuration."
            )
            config = None
        
        except json.JSONDecodeError as e:
            logger.warning(
                f"JSON parsing error in {config_path}: {str(e)}. "
                "Falling back to default configuration."
            )
            config = None
        
        except Exception as e:
            logger.warning(
                f"Error reading configuration from {config_path}: {str(e)}. "
                "Falling back to default configuration."
            )
            config = None
    else:
        if config_path != Path('config/model_params.yml'):
            # Only log as warning if explicit path provided
            logger.warning(
                f"Configuration file not found: {config_path}. "
                "Using default configuration."
            )
        else:
            logger.debug(
                "Default config file not found at config/model_params.yml. "
                "Using hardcoded defaults."
            )
    
    # Use defaults if no config loaded
    if config is None:
        config = get_default_config()
        logger.info("Using default configuration")
    
    # Validate configuration
    try:
        if not validate_config(config):
            raise ConfigurationError("Configuration validation failed")
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise
    
    logger.info("Configuration loaded and validated successfully")
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration schema and parameter ranges.
    
    Validates that all required sections are present and that parameter values
    fall within acceptable ranges. Per Architecture Section 7, enforces:
    - Range validation for all numeric parameters
    - Required section existence
    - Type checking for specific parameters
    - Dual-space processing constraints
    
    Validation Rules:
    - max_p, max_d, max_q: Must be > 0 (ARIMA parameters)
    - nodes: Must be in range 5-20 (LSTM neurons)
    - dropout_rate: Must be in [0.0, 1.0] (regularization)
    - batch_size, epochs, early_stopping_patience, window_size: Must be > 0
    - test_size: Must be in [0.1, 0.5] (validation ratio)
    - gpu_memory_fraction: Must be in [0.1, 1.0]
    - Required sections: arima, lstm, validation, hardware, output
    - output must have all dual-space flags
    
    Args:
        config (dict): Configuration dictionary to validate.
    
    Returns:
        bool: True if configuration is valid.
    
    Raises:
        ConfigurationError: If validation fails with details about the error.
    
    Examples:
        >>> config = get_default_config()
        >>> validate_config(config)  # Returns True for valid config
        True
        >>> 
        >>> bad_config = get_default_config()
        >>> bad_config['lstm']['nodes'] = 25  # Out of range
        >>> validate_config(bad_config)  # Raises ConfigurationError
        Traceback (most recent call last):
        ...
        ConfigurationError: LSTM nodes must be between 5 and 20, got 25
    """
    logger.debug("Validating configuration...")
    
    try:
        # ===== Check Required Sections =====
        required_sections = ['arima', 'lstm', 'validation', 'hardware', 'output']
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required section: {section}")
        
        # ===== ARIMA Validation (returns space) =====
        arima = config['arima']
        
        if not isinstance(arima.get('max_p'), int) or arima['max_p'] <= 0:
            raise ConfigurationError(
                f"ARIMA max_p must be positive integer, got {arima.get('max_p')}"
            )
        
        if not isinstance(arima.get('max_d'), int) or arima['max_d'] <= 0:
            raise ConfigurationError(
                f"ARIMA max_d must be positive integer, got {arima.get('max_d')}"
            )
        
        if not isinstance(arima.get('max_q'), int) or arima['max_q'] <= 0:
            raise ConfigurationError(
                f"ARIMA max_q must be positive integer, got {arima.get('max_q')}"
            )
        
        if arima.get('information_criterion') not in ['aic', 'bic']:
            raise ConfigurationError(
                f"ARIMA information_criterion must be 'aic' or 'bic', "
                f"got {arima.get('information_criterion')}"
            )
        
        # ===== LSTM Validation (returns space residuals) =====
        lstm = config['lstm']
        
        if not isinstance(lstm.get('nodes'), (int, float)) or lstm['nodes'] < 5 or lstm['nodes'] > 20:
            raise ConfigurationError(
                f"LSTM nodes must be between 5 and 20, got {lstm.get('nodes')}"
            )
        
        if not isinstance(lstm.get('dropout_rate'), (int, float)):
            raise ConfigurationError(
                f"LSTM dropout_rate must be numeric, got {type(lstm.get('dropout_rate'))}"
            )
        
        if lstm['dropout_rate'] < 0.0 or lstm['dropout_rate'] > 1.0:
            raise ConfigurationError(
                f"LSTM dropout_rate must be between 0.0 and 1.0, "
                f"got {lstm['dropout_rate']}"
            )
        
        if not isinstance(lstm.get('batch_size'), int) or lstm['batch_size'] <= 0:
            raise ConfigurationError(
                f"LSTM batch_size must be positive integer, got {lstm.get('batch_size')}"
            )
        
        if not isinstance(lstm.get('epochs'), int) or lstm['epochs'] <= 0:
            raise ConfigurationError(
                f"LSTM epochs must be positive integer, got {lstm.get('epochs')}"
            )
        
        if not isinstance(lstm.get('early_stopping_patience'), int) or lstm['early_stopping_patience'] <= 0:
            raise ConfigurationError(
                f"LSTM early_stopping_patience must be positive integer, "
                f"got {lstm.get('early_stopping_patience')}"
            )
        
        if not isinstance(lstm.get('window_size'), int) or lstm['window_size'] <= 0:
            raise ConfigurationError(
                f"LSTM window_size must be positive integer, got {lstm.get('window_size')}"
            )
        
        if lstm.get('optimizer') not in ['adam', 'sgd', 'rmsprop']:
            raise ConfigurationError(
                f"LSTM optimizer must be 'adam', 'sgd', or 'rmsprop', "
                f"got {lstm.get('optimizer')}"
            )
        
        # ===== Validation Configuration =====
        validation = config['validation']
        
        if validation.get('method') not in ['walk_forward', 'time_series_split']:
            raise ConfigurationError(
                f"Validation method must be 'walk_forward' or 'time_series_split', "
                f"got {validation.get('method')}"
            )
        
        test_size = validation.get('test_size')
        if not isinstance(test_size, (int, float)) or test_size < 0.1 or test_size > 0.5:
            raise ConfigurationError(
                f"Validation test_size must be between 0.1 and 0.5, got {test_size}"
            )
        
        # ===== Hardware Configuration =====
        hardware = config['hardware']
        
        gpu_mem = hardware.get('gpu_memory_fraction')
        if not isinstance(gpu_mem, (int, float)) or gpu_mem < 0.1 or gpu_mem > 1.0:
            raise ConfigurationError(
                f"Hardware gpu_memory_fraction must be between 0.1 and 1.0, "
                f"got {gpu_mem}"
            )
        
        # ===== Output Configuration (dual-space) =====
        output = config['output']
        
        required_output_flags = [
            'include_returns_space',
            'include_price_space',
            'metrics_in_both_spaces'
        ]
        
        for flag in required_output_flags:
            if flag not in output:
                raise ConfigurationError(f"Output configuration missing: {flag}")
            
            if not isinstance(output[flag], bool):
                raise ConfigurationError(
                    f"Output {flag} must be boolean, got {type(output[flag])}"
                )
        
        # Dual-space constraint: both spaces must be included (per architecture)
        if not output['include_returns_space'] or not output['include_price_space']:
            raise ConfigurationError(
                "Both returns_space and price_space must be enabled "
                "(dual-space processing constraint)"
            )
        
        logger.info("Configuration validation successful")
        logger.debug(f"Config summary: ARIMA order limiting to "
                    f"p≤{arima['max_p']}, d≤{arima['max_d']}, q≤{arima['max_q']}; "
                    f"LSTM with {lstm['nodes']} nodes, "
                    f"dropout={lstm['dropout_rate']}, "
                    f"window_size={lstm['window_size']}")
        
        return True
    
    except ConfigurationError:
        raise
    except Exception as e:
        error_msg = f"Unexpected error during configuration validation: {str(e)}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg)


def merge_config(
    base_config: Dict[str, Any],
    overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge configuration overrides into base configuration.
    
    Performs deep merge of override parameters into base configuration,
    enabling CLI or programmatic parameter overrides without modifying
    the base configuration file. Useful for:
    - CLI parameter overrides (--lstm-nodes 20)
    - Experiment variations without code changes
    - Parameter sweeps and automated testing
    
    Merge Behavior:
    - Deep merge for nested dictionaries (only overrides specified values)
    - Preserves all base values not in overrides
    - Logs which values were overridden
    - Validates merged configuration before returning
    - Overwrites primitive values completely (lists, scalars)
    
    Args:
        base_config (dict): Base configuration dictionary (should be pre-validated).
        overrides (dict): Override parameters (nested dict or flat dict).
                         Examples:
                         - {'lstm': {'nodes': 20}}  # Nested
                         - {'lstm.nodes': 20}       # Flat (future extension)
    
    Returns:
        dict: Merged configuration with overrides applied.
    
    Raises:
        ConfigurationError: If merged configuration fails validation.
    
    Examples:
        >>> base = get_default_config()
        >>> base['lstm']['nodes']
        10
        >>> 
        >>> overrides = {'lstm': {'nodes': 20}}
        >>> merged = merge_config(base, overrides)
        >>> merged['lstm']['nodes']
        20
        >>> 
        >>> # Other LSTM params preserved
        >>> merged['lstm']['batch_size']
        64
    """
    import copy
    
    logger.debug("Merging configuration overrides...")
    
    # Deep copy base to avoid modifying original
    merged = copy.deepcopy(base_config)
    
    # Perform deep merge
    def deep_merge(target: Dict, source: Dict, path: str = "") -> None:
        """Recursively merge source into target, logging changes."""
        for key, value in source.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in target:
                logger.warning(f"Override key not in base config: {current_path}")
                target[key] = value
            elif isinstance(value, dict) and isinstance(target.get(key), dict):
                # Recursive merge for nested dicts
                deep_merge(target[key], value, current_path)
            else:
                # Override primitive value
                old_value = target[key]
                target[key] = value
                logger.info(f"Override config: {current_path} = {value} (was {old_value})")
    
    # Apply overrides
    if overrides:
        deep_merge(merged, overrides)
    
    # Validate merged configuration
    try:
        validate_config(merged)
        logger.info("Merged configuration validated successfully")
    except ConfigurationError as e:
        logger.error(f"Merged configuration validation failed: {str(e)}")
        raise
    
    return merged


def config_to_dict(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert nested configuration to flat dictionary for display.
    
    Flattens hierarchical configuration structure into a single-level dictionary
    with dot-separated keys. Useful for:
    - Logging configuration state
    - CLI display and debugging
    - Saving to simple formats (env vars, flat JSON)
    
    Key Format:
    - Nested keys become dot-separated: 'lstm.nodes', 'arima.max_p'
    - Values converted to strings for display
    - Special formatting for booleans and numeric values
    
    Args:
        config (dict): Configuration dictionary (nested).
    
    Returns:
        dict: Flat dictionary with dot-separated keys and string values.
    
    Examples:
        >>> config = get_default_config()
        >>> flat = config_to_dict(config)
        >>> flat['lstm.nodes']
        '10'
        >>> flat['arima.max_p']
        '5'
        >>> 
        >>> # Useful for logging
        >>> for key, value in sorted(flat.items()):
        ...     print(f"{key}: {value}")
        arima.information_criterion: aic
        arima.max_d: 2
        arima.max_p: 5
        arima.max_q: 5
        ...
    """
    flat = {}
    
    def flatten(d: Dict, parent_key: str = "") -> None:
        """Recursively flatten nested dictionary."""
        for key, value in d.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                flatten(value, new_key)
            else:
                # Convert to string for display
                if isinstance(value, bool):
                    flat[new_key] = str(value).lower()  # 'true' or 'false'
                else:
                    flat[new_key] = str(value)
    
    flatten(config)
    logger.debug(f"Configuration flattened to {len(flat)} keys")
    
    return flat
