### src/models/predictor.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from src.config import RISK_THRESHOLDS
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StudentPredictor:
    """Make predictions and classify risk levels"""
    
    def __init__(self, model):
        self.model = model
        
    def predict_outcomes(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict pass/fail outcomes
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        logger.info(f"Making predictions for {len(X)} students...")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        pass_count = predictions.sum()
        fail_count = len(predictions) - pass_count
        
        logger.info(f"Predictions: {pass_count} Pass, {fail_count} Fail")
        
        return predictions, probabilities
    
    def classify_risk_level(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Classify students into risk levels based on pass probability
        
        Args:
            probabilities: Predicted probabilities of passing
            
        Returns:
            Array of risk levels ('Low', 'Medium', 'High')
        """
        logger.info("Classifying risk levels...")
        
        risk_levels = np.empty(len(probabilities), dtype=object)
        
        # High risk: probability < 50%
        high_risk_mask = probabilities < RISK_THRESHOLDS['high']
        risk_levels[high_risk_mask] = 'High'
        
        # Medium risk: 50% <= probability < 70%
        medium_risk_mask = (probabilities >= RISK_THRESHOLDS['high']) & \
                          (probabilities < RISK_THRESHOLDS['medium'])
        risk_levels[medium_risk_mask] = 'Medium'
        
        # Low risk: probability >= 70%
        low_risk_mask = probabilities >= RISK_THRESHOLDS['low']
        risk_levels[low_risk_mask] = 'Low'
        
        # Count by risk level
        unique, counts = np.unique(risk_levels, return_counts=True)
        risk_distribution = dict(zip(unique, counts))
        
        logger.info(f"Risk distribution:")
        for risk, count in risk_distribution.items():
            logger.info(f"  {risk}: {count} ({count/len(probabilities)*100:.1f}%)")
        
        return risk_levels
    
    def create_prediction_report(self, X: pd.DataFrame, 
                                student_ids: pd.Series = None) -> pd.DataFrame:
        """
        Create detailed prediction report
        
        Args:
            X: Feature matrix
            student_ids: Student IDs (optional)
            
        Returns:
            DataFrame with predictions and risk levels
        """
        logger.info("Creating prediction report...")
        
        predictions, probabilities = self.predict_outcomes(X)
        risk_levels = self.classify_risk_level(probabilities)
        
        report_df = pd.DataFrame({
            'prediction': predictions,
            'pass_probability': probabilities,
            'fail_probability': 1 - probabilities,
            'risk_level': risk_levels,
            'predicted_outcome': ['Pass' if p == 1 else 'Fail' for p in predictions]
        })
        
        if student_ids is not None:
            report_df.insert(0, 'student_id', student_ids.values)
        
        # Add key features if available
        if hasattr(self.model, 'feature_importances_'):
            top_features = X.columns[np.argsort(self.model.feature_importances_)[-5:]]
            for feat in top_features:
                if feat in X.columns:
                    report_df[f'feature_{feat}'] = X[feat].values
        
        logger.info(f"Prediction report created: {len(report_df)} records")
        
        return report_df
    
    def get_at_risk_students(self, prediction_report: pd.DataFrame,
                            risk_levels: list = ['High', 'Medium']) -> pd.DataFrame:
        """
        Get list of at-risk students
        
        Args:
            prediction_report: Prediction report DataFrame
            risk_levels: List of risk levels to include
            
        Returns:
            DataFrame with at-risk students
        """
        logger.info(f"Filtering at-risk students (levels: {risk_levels})...")
        
        at_risk = prediction_report[prediction_report['risk_level'].isin(risk_levels)].copy()
        
        # Sort by risk level and probability
        risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
        at_risk['risk_order'] = at_risk['risk_level'].map(risk_order)
        at_risk = at_risk.sort_values(['risk_order', 'pass_probability'])
        at_risk = at_risk.drop('risk_order', axis=1)
        
        logger.info(f"Found {len(at_risk)} at-risk students")
        
        return at_risk