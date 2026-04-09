### src/models/improved_models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict
from src.config import MODEL_PARAMS
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RandomForestModel:
    """Random Forest Classifier"""
    
    def __init__(self, **kwargs):
        params = MODEL_PARAMS['random_forest'].copy()
        params.update(kwargs)
        self.model = RandomForestClassifier(**params)
        self.is_trained = False
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the Random Forest model"""
        logger.info("Training Random Forest model...")
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        logger.info(f"Model trained successfully")
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("Evaluating Random Forest model...")
        
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
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_model(self):
        """Get the trained model"""
        return self.model


class GradientBoostingModel:
    """Gradient Boosting Classifier"""
    
    def __init__(self, **kwargs):
        params = MODEL_PARAMS['gradient_boosting'].copy()
        params.update(kwargs)
        self.model = GradientBoostingClassifier(**params)
        self.is_trained = False
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the Gradient Boosting model"""
        logger.info("Training Gradient Boosting model...")
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        logger.info(f"Model trained successfully")
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("Evaluating Gradient Boosting model...")
        
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
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_model(self):
        """Get the trained model"""
        return self.model