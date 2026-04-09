### src/models/baseline.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Tuple
from src.config import MODEL_PARAMS, RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaselineModel:
    """Logistic Regression baseline model"""
    
    def __init__(self):
        self.model = LogisticRegression(**MODEL_PARAMS['logistic_regression'])
        self.is_trained = False
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the baseline model
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info("Training Logistic Regression baseline model...")
        
        self.feature_names = X_train.columns.tolist()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        logger.info(f"Model trained successfully")
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability estimates
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating baseline model...")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        logger.info(f"Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (coefficients)
        
        Returns:
            DataFrame with features and their importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        })
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def get_model(self):
        """Get the trained model"""
        return self.model