"""
Custom Exception Classes for Hybrid LSTM-ARIMA Forecasting System

Implements a custom exception hierarchy for centralized error handling across
the entire system. Each exception includes descriptive attributes for logging
and recovery strategy determination.

Section 10: Error Handling Strategy
"""


class DataValidationError(Exception):
    """
    Raised when invalid input data is encountered.
    
    Triggered by:
    - Missing files
    - Wrong data format
    - Malformed data structures
    
    Example: "No price data found in file"
    """
    
    def __init__(self, error_message, file_path=None, data_shape=None):
        """
        Initialize DataValidationError.
        
        Args:
            error_message (str): Description of the validation error
            file_path (str, optional): Path to the problematic file
            data_shape (tuple, optional): Expected vs actual shape info
        """
        super().__init__(error_message)
        self.error_message = error_message
        self.file_path = file_path
        self.data_shape = data_shape
    
    def __str__(self):
        msg = f"Data Validation Error: {self.error_message}"
        if self.file_path:
            msg += f"\n  File: {self.file_path}"
        if self.data_shape:
            msg += f"\n  Data shape: {self.data_shape}"
        return msg


class ModelConvergenceError(Exception):
    """
    Raised when ARIMA fails to converge or fitting fails.
    
    Triggered by:
    - ARIMA non-convergence
    - Fitting failures after max iterations
    
    Example: "ARIMA(5,2,5) failed to converge after 100 iterations"
    
    Recovery: Log warning, use default params (1,1,1) for ARIMA
    """
    
    def __init__(self, error_message, model_type=None, parameters=None):
        """
        Initialize ModelConvergenceError.
        
        Args:
            error_message (str): Description of convergence failure
            model_type (str, optional): Type of model (e.g., "ARIMA", "LSTM")
            parameters (tuple/dict, optional): Failed model parameters
        """
        super().__init__(error_message)
        self.error_message = error_message
        self.model_type = model_type
        self.parameters = parameters
    
    def __str__(self):
        msg = f"Model Convergence Error ({self.model_type}): {self.error_message}"
        if self.parameters:
            msg += f"\n  Parameters: {self.parameters}"
        return msg


class FileIOError(Exception):
    """
    Raised when file reading/writing problems occur.
    
    Triggered by:
    - File not found
    - Permission denied
    - Disk space issues
    - Encoding errors
    
    Example: "Cannot write to output/forecast.csv - permission denied"
    
    Recovery: Retry up to 2 times, then exit with error
    """
    
    def __init__(self, error_message, file_path=None, operation=None):
        """
        Initialize FileIOError.
        
        Args:
            error_message (str): Description of I/O error
            file_path (str, optional): Path to the file causing issues
            operation (str, optional): Operation type ("read" or "write")
        """
        super().__init__(error_message)
        self.error_message = error_message
        self.file_path = file_path
        self.operation = operation
    
    def __str__(self):
        msg = f"File I/O Error ({self.operation}): {self.error_message}"
        if self.file_path:
            msg += f"\n  File: {self.file_path}"
        return msg


class ConfigurationError(Exception):
    """
    Raised when invalid configuration parameters are provided.
    
    Triggered by:
    - Parameter values out of allowed range
    - Invalid parameter combinations
    - Missing required configuration
    
    Example: "LSTM nodes=25 exceeds max value of 20"
    
    Recovery: Exit with valid parameter ranges displayed
    """
    
    def __init__(self, error_message, parameter_name=None, invalid_value=None, allowed_range=None):
        """
        Initialize ConfigurationError.
        
        Args:
            error_message (str): Description of configuration error
            parameter_name (str, optional): Name of invalid parameter
            invalid_value (any, optional): Value that failed validation
            allowed_range (str/tuple, optional): Valid range or allowed values
        """
        super().__init__(error_message)
        self.error_message = error_message
        self.parameter_name = parameter_name
        self.invalid_value = invalid_value
        self.allowed_range = allowed_range
    
    def __str__(self):
        msg = f"Configuration Error: {self.error_message}"
        if self.parameter_name:
            msg += f"\n  Parameter: {self.parameter_name}"
        if self.invalid_value is not None:
            msg += f"\n  Invalid value: {self.invalid_value}"
        if self.allowed_range:
            msg += f"\n  Allowed range: {self.allowed_range}"
        return msg


class PriceConversionError(Exception):
    """
    Raised when price reconstruction from returns fails.
    
    Triggered by:
    - Negative price generation
    - Invalid price reconstruction
    - Loss of precision
    
    Example: "Price reconstruction resulted in negative prices"
    
    Recovery: Log warning, return returns-space forecast only (no price-space)
    """
    
    def __init__(self, error_message, prices=None, formula=None):
        """
        Initialize PriceConversionError.
        
        Args:
            error_message (str): Description of conversion failure
            prices (array-like, optional): Problematic price values
            formula (str, optional): Formula used for conversion
        """
        super().__init__(error_message)
        self.error_message = error_message
        self.prices = prices
        self.formula = formula
    
    def __str__(self):
        msg = f"Price Conversion Error: {self.error_message}"
        if self.formula:
            msg += f"\n  Formula: {self.formula}"
        if self.prices is not None:
            try:
                min_price = min(self.prices) if self.prices is not None else "N/A"
                max_price = max(self.prices) if self.prices is not None else "N/A"
                msg += f"\n  Price range: [{min_price}, {max_price}]"
            except (TypeError, ValueError):
                msg += f"\n  Prices: {self.prices}"
        return msg
