### src/models/trainer.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    """Generic model trainer with logging and tracking"""
    
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.training_history = {}
        self.is_trained = False
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Train model and track performance
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Training history
        """
        logger.info(f"Training {self.model_name}...")
        
        start_time = datetime.now()
        
        # Train model
        self.model.train(X_train, y_train)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Evaluate on training set
        train_metrics = self.model.evaluate(X_train, y_train)
        
        # Evaluate on validation set if provided
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_metrics = self.model.evaluate(X_val, y_val)
        
        # Store training history
        self.training_history = {
            'model_name': self.model_name,
            'training_time_seconds': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'n_train_samples': len(X_train),
            'n_features': X_train.shape[1],
            'timestamp': datetime.now().isoformat()
        }
        
        self.is_trained = True
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Training metrics: {train_metrics}")
        if val_metrics:
            logger.info(f"Validation metrics: {val_metrics}")
        
        return self.training_history
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history"""
        return self.training_history
    
    def get_model(self):
        """Get the trained model"""
        return self.model