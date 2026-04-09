### src/evaluation/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from typing import Dict, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate all classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating evaluation metrics...")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        self.metrics = metrics
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate)"""
        cm = confusion_matrix(y_true, y_pred)
        tn = cm[0, 0]
        fp = cm[0, 1]
        
        if (tn + fp) == 0:
            return 0.0
        
        return tn / (tn + fp)
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 target_names: list = None) -> str:
        """Get detailed classification report"""
        if target_names is None:
            target_names = ['Fail', 'Pass']
        
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def get_roc_curve_data(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple:
        """
        Get ROC curve data
        
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        return roc_curve(y_true, y_proba)
    
    def get_precision_recall_curve_data(self, y_true: np.ndarray, 
                                       y_proba: np.ndarray) -> Tuple:
        """
        Get Precision-Recall curve data
        
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        return precision_recall_curve(y_true, y_proba)
    
    def compare_models(self, models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models_results: Dictionary with model names as keys and metrics as values
            
        Returns:
            DataFrame with comparison
        """
        logger.info("Comparing models...")
        
        comparison_df = pd.DataFrame(models_results).T
        comparison_df = comparison_df.round(4)
        
        # Add rank for each metric
        for col in comparison_df.columns:
            comparison_df[f'{col}_rank'] = comparison_df[col].rank(ascending=False)
        
        return comparison_df