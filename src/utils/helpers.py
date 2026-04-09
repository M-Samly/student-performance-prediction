### src/utils/helpers.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric column names"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of categorical column names"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def calculate_missing_percentage(df: pd.DataFrame) -> pd.Series:
    """Calculate percentage of missing values per column"""
    return (df.isnull().sum() / len(df) * 100).round(2)

def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print comprehensive DataFrame information"""
    logger.info(f"\n{name} Information:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    missing_pct = calculate_missing_percentage(df)
    if missing_pct.sum() > 0:
        logger.info(f"\nMissing values:")
        logger.info(missing_pct[missing_pct > 0].to_string())

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string"""
    return f"{value:.{decimals}f}%"

def create_bins(data: pd.Series, n_bins: int = 4, labels: List[str] = None) -> pd.Series:
    """Create bins from continuous data"""
    try:
        return pd.cut(data, bins=n_bins, labels=labels)
    except Exception as e:
        logger.error(f"Error creating bins: {e}")
        return data

def get_top_n_items(data: pd.Series, n: int = 10, ascending: bool = False) -> pd.Series:
    """Get top N items from a Series"""
    return data.nlargest(n) if not ascending else data.nsmallest(n)