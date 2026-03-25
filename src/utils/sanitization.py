"""JSON sanitization utilities for handling special float values and numpy types."""
import math
from typing import Any, Dict, Optional, Union

# Try to import numpy, but make it optional
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def sanitize_value(value: Any) -> Any:
    """
    Convert numpy types and special float values to JSON-serializable Python types.
    
    Args:
        value: Any value that may need conversion
        
    Returns:
        JSON-serializable Python native type
    """
    if value is None:
        return None
    
    # Handle numpy types
    if HAS_NUMPY:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            if np.isnan(value) or np.isinf(value):
                return None
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.bool_):
            return bool(value)
    
    # Handle Python float NaN/Infinity
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    
    return value


def sanitize_float(value: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """
    Convert NaN/Infinity to None for JSON serialization.
    
    JSON does not support NaN or Infinity values, so we convert them to None
    to avoid serialization errors.
    
    Args:
        value: A numeric value that may be NaN or Infinity
        
    Returns:
        The original value if valid, or None if NaN/Infinity
    """
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def sanitize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize all values in a dictionary for JSON serialization.
    
    Converts numpy types and NaN/Infinity values to JSON-serializable types.
    
    Args:
        record: A dictionary that may contain non-serializable values
        
    Returns:
        A new dictionary with sanitized values
    """
    return {k: sanitize_value(v) for k, v in record.items()}


def sanitize_records(records: list) -> list:
    """
    Sanitize all records in a list.
    
    Args:
        records: A list of dictionaries to sanitize
        
    Returns:
        A new list with all records sanitized
    """
    return [sanitize_record(r) for r in records]
