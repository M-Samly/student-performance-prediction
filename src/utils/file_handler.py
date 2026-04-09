### src/utils/file_handler.py
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Any, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def load_csv(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Load CSV file into pandas DataFrame
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
    
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        logger.info(f"Loading CSV from: {file_path}")
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV from {file_path}: {str(e)}")
        raise


def save_csv(df: pd.DataFrame, file_path: Path, **kwargs) -> None:
    """
    Save DataFrame to CSV file
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        **kwargs: Additional arguments for df.to_csv
    """
    try:
        logger.info(f"Saving CSV to: {file_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False, **kwargs)
        logger.info(f"Successfully saved {len(df)} rows to {file_path}")
    except Exception as e:
        logger.error(f"Error saving CSV to {file_path}: {str(e)}")
        raise


def save_model(model: Any, file_path: Path, metadata: Dict = None) -> None:
    """
    Save machine learning model
    
    Args:
        model: Model object to save
        file_path: Output file path
        metadata: Optional metadata dictionary
    """
    try:
        logger.info(f"Saving model to: {file_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, file_path)
        
        # Save metadata if provided
        if metadata:
            metadata_path = file_path.with_suffix('.json')
            save_json(metadata, metadata_path)
        
        logger.info(f"Successfully saved model to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {str(e)}")
        raise


def load_model(file_path: Path) -> Any:
    """
    Load machine learning model
    
    Args:
        file_path: Path to model file
    
    Returns:
        Loaded model object
    """
    try:
        logger.info(f"Loading model from: {file_path}")
        model = joblib.load(file_path)
        logger.info(f"Successfully loaded model from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {str(e)}")
        raise


def save_json(data: Dict, file_path: Path) -> None:
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    try:
        logger.info(f"Saving JSON to: {file_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types
        converted_data = convert_numpy_types(data)
        
        with open(file_path, 'w') as f:
            json.dump(converted_data, f, indent=4, cls=NumpyEncoder)
        
        logger.info(f"Successfully saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        raise


def load_json(file_path: Path) -> Dict:
    """
    Load JSON file into dictionary
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Dictionary with loaded data
    """
    try:
        logger.info(f"Loading JSON from: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        raise