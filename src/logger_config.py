"""
Logging Configuration for Hybrid LSTM-ARIMA Forecasting System

Provides centralized logging setup with module-specific loggers.
Supports both file and console output with standardized formatting.

Section 10: Error Handling Strategy - Logging Configuration
"""

import logging
import logging.handlers
import traceback
from datetime import datetime, timezone
from pathlib import Path


# Global logger instance
_root_logger = None
_logger_instances = {}


class UTCFormatter(logging.Formatter):
    """Custom formatter with UTC ISO 8601 timestamps and color support."""
    
    # ANSI color codes for console output
    _COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    _RESET = '\033[0m'           # Reset color
    
    def format(self, record):
        """
        Format log record with UTC timestamp and optional color.
        
        Format: [TIMESTAMP] [LEVEL] [MODULE] - MESSAGE
        Example: [2026-01-15T18:48:45.262Z] [INFO] [arima_engine] - ARIMA model fitted with order (1,1,0)
        """
        # Create UTC datetime with ISO 8601 format
        timestamp = datetime.fromtimestamp(
            record.created,
            tz=timezone.utc
        ).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        
        # Extract module name from logger name (e.g., "src.preprocessing" -> "preprocessing")
        module_parts = record.name.split('.')
        module_name = module_parts[-1] if module_parts else "root"
        
        # Build base message
        base_message = f"[{timestamp}] [{record.levelname}] [{module_name}] - {record.getMessage()}"
        
        # Add color for console output if not a file handler
        if isinstance(self._style, logging.PercentStyle):
            return base_message
        
        return base_message
    
    def format_for_console(self, record):
        """Version of format that includes colors for terminal output."""
        timestamp = datetime.fromtimestamp(
            record.created,
            tz=timezone.utc
        ).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        
        module_parts = record.name.split('.')
        module_name = module_parts[-1] if module_parts else "root"
        
        level_color = self._COLORS.get(record.levelname, '')
        
        message = (
            f"[{timestamp}] "
            f"{level_color}[{record.levelname}]{self._RESET} "
            f"[{module_name}] - {record.getMessage()}"
        )
        
        return message


class ColoredConsoleHandler(logging.StreamHandler):
    """Console handler with color support."""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)
    
    def format(self, record):
        """Use colored format for console output."""
        formatter = self.formatter
        if hasattr(formatter, 'format_for_console'):
            return formatter.format_for_console(record)
        return super().format(record)


def configure_logging(log_level='INFO', log_file=None):
    """
    Setup centralized logging configuration.
    
    Configures root logger with optional file handler and console handler.
    Sets up both file and console output with standardized format.
    
    Args:
        log_level (str): Logging level - DEBUG, INFO, WARNING, ERROR, CRITICAL
                        Default: 'INFO'
        log_file (str, optional): Path to log file. If provided, logs to both
                                 file and console. If None, logs to console only.
    
    Returns:
        logging.Logger: Configured root logger instance
    
    Raises:
        ValueError: If log_level is invalid
        
    Example:
        >>> logger = configure_logging(log_level='DEBUG', log_file='output/run.log')
        >>> logger.info("System initialized")
    """
    global _root_logger
    
    # Validate log level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if log_level.upper() not in valid_levels:
        raise ValueError(
            f"Invalid log_level '{log_level}'. Must be one of: {', '.join(valid_levels)}"
        )
    
    # Get or create root logger
    _root_logger = logging.getLogger()
    _root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    _root_logger.handlers = []
    
    # Create formatter (UTCFormatter with ISO 8601)
    formatter = UTCFormatter()
    
    # Add console handler (with colors)
    console_handler = ColoredConsoleHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    _root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler (no colors in file)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        ))
        _root_logger.addHandler(file_handler)
        
        _root_logger.info(f"Logging configured: level={log_level}, file={log_file}")
    else:
        _root_logger.info(f"Logging configured: level={log_level}, console only")
    
    return _root_logger


def get_logger(module_name):
    """
    Get a module-specific logger configured with this system's format.
    
    Each module should call this function at module level:
        logger = get_logger(__name__)
    
    Args:
        module_name (str): Name of the module (typically __name__)
                          Example: "src.arima_engine", "src.preprocessing"
    
    Returns:
        logging.Logger: Logger instance configured for the module
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting ARIMA fitting")
        [2026-01-15T18:48:45.262Z] [INFO] [arima_engine] - Starting ARIMA fitting
    """
    global _logger_instances
    
    # Return cached logger if already created
    if module_name in _logger_instances:
        return _logger_instances[module_name]
    
    # Create new logger for module
    logger = logging.getLogger(module_name)
    
    # Set to use root logger's handlers if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = True
    
    # Cache the logger instance
    _logger_instances[module_name] = logger
    
    return logger


def log_exception(logger, exception):
    """
    Log full exception details including traceback.
    
    Logs exception type, message, and full traceback at ERROR level.
    Useful for debugging and error tracking.
    
    Args:
        logger (logging.Logger): Logger instance to use
        exception (Exception): Exception object to log
    
    Returns:
        None
    
    Example:
        >>> try:
        >>>     data = load_data(file_path)
        >>> except Exception as e:
        >>>     from src.logger_config import log_exception
        >>>     log_exception(logger, e)
        >>>     raise
    """
    # Get full traceback as string
    tb_str = traceback.format_exc()
    
    # Log with full context
    logger.error(
        f"Exception occurred: {type(exception).__name__}\n"
        f"Message: {str(exception)}\n"
        f"Traceback:\n{tb_str}"
    )


# Convenience function for one-shot setup
def setup_logging_simple(log_level='INFO', log_file=None):
    """
    Simple one-shot logging setup function.
    
    Alias for configure_logging for convenience.
    
    Args:
        log_level (str): Logging level
        log_file (str, optional): Optional log file path
    
    Returns:
        logging.Logger: Configured root logger
    """
    return configure_logging(log_level, log_file)
