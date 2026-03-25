"""JSON sanitization utilities for handling special float values."""
import math
from typing import Any, Dict, Optional, Union


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
    Sanitize all float values in a dictionary for JSON serialization.
    
    Recursively processes dictionary values and converts any NaN/Infinity
    float values to None.
    
    Args:
        record: A dictionary that may contain NaN/Infinity values
        
    Returns:
        A new dictionary with sanitized values
    """
    return {
        k: sanitize_float(v) if isinstance(v, (float, int, type(None))) else v
        for k, v in record.items()
    }


def sanitize_records(records: list) -> list:
    """
    Sanitize all records in a list.
    
    Args:
        records: A list of dictionaries to sanitize
        
    Returns:
        A new list with all records sanitized
    """
    return [sanitize_record(r) for r in records]
