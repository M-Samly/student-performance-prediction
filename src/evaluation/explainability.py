### src/evaluation/explainability.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Any, List
from pathlib import Path
from src.config import FIGURES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelExplainer:
    """Model explainability using SHAP and other techniques"""
    
    def __init__(self, model: Any, X_train: pd.DataFrame, save_dir: Path = FIGURES_DIR):
        self.model = model
        self.X_train = X_train
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.explainer = None
        self.shap_values = None
        
    def create_shap_explainer(self, model_type: str = 'tree') -> None:
        """
        Create SHAP explainer
        
        Args:
            model_type: 'tree' for tree-based models, 'linear' for linear models
        """
        logger.info(f"Creating SHAP explainer (type: {model_type})...")
        
        try:
            if model_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
            elif model_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
            else:
                self.explainer = shap.Explainer(self.model, self.X_train)
            
            logger.info("SHAP explainer created successfully")
        except Exception as e:
            logger.warning(f"Could not create SHAP explainer: {str(e)}")
            logger.warning("SHAP analysis will be skipped")
    
    def calculate_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values"""
        if self.explainer is None:
            logger.warning("SHAP explainer not initialized")
            return None
        
        logger.info("Calculating SHAP values...")
        
        try:
            self.shap_values = self.explainer.shap_values(X)
            logger.info("SHAP values calculated successfully")
            return self.shap_values
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            return None
    
    def plot_shap_summary(self, X: pd.DataFrame, 
                         save_name: str = "shap_summary.png") -> None:
        """Plot SHAP summary plot"""
        if self.explainer is None:
            logger.warning("Cannot plot SHAP summary: explainer not initialized")
            return
        
        logger.info("Creating SHAP summary plot...")
        
        try:
            if self.shap_values is None:
                self.calculate_shap_values(X)
            
            # For binary classification, use the positive class
            shap_vals = self.shap_values
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_vals, X, show=False)
            plt.tight_layout()
            
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP summary plot saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {str(e)}")
    
    def plot_shap_importance(self, X: pd.DataFrame, top_n: int = 15,
                            save_name: str = "shap_importance.png") -> None:
        """Plot SHAP feature importance"""
        if self.explainer is None:
            logger.warning("Cannot plot SHAP importance: explainer not initialized")
            return
        
        logger.info("Creating SHAP importance plot...")
        
        try:
            if self.shap_values is None:
                self.calculate_shap_values(X)
            
            shap_vals = self.shap_values
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            
            # Calculate mean absolute SHAP values
            shap_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(shap_vals).mean(axis=0)
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(shap_importance)), shap_importance['importance'], color='steelblue')
            plt.yticks(range(len(shap_importance)), shap_importance['feature'])
            plt.xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
            plt.ylabel('Feature', fontsize=12, fontweight='bold')
            plt.title('SHAP Feature Importance', fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP importance plot saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error creating SHAP importance plot: {str(e)}")
    
    def get_top_shap_features(self, X: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Get top features by SHAP importance"""
        if self.explainer is None or self.shap_values is None:
            logger.warning("Cannot get SHAP features: values not calculated")
            return None
        
        try:
            shap_vals = self.shap_values
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'shap_importance': np.abs(shap_vals).mean(axis=0)
            }).sort_values('shap_importance', ascending=False).head(top_n)
            
            return importance_df
        except Exception as e:
            logger.error(f"Error getting top SHAP features: {str(e)}")
            return None