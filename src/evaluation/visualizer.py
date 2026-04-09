### src/evaluation/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from pathlib import Path
from typing import List, Dict
from src.config import FIGURES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelVisualizer:
    """Visualization for model evaluation"""
    
    def __init__(self, save_dir: Path = FIGURES_DIR):
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             title: str = "Confusion Matrix",
                             save_name: str = "confusion_matrix.png") -> None:
        """Plot confusion matrix"""
        logger.info(f"Plotting confusion matrix: {title}")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fail', 'Pass'],
                   yticklabels=['Fail', 'Pass'],
                   cbar_kws={'label': 'Count'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.7, f'({cm[i,j]/total*100:.1f}%)',
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      model_name: str = "Model",
                      save_name: str = "roc_curve.png") -> None:
        """Plot ROC curve"""
        logger.info(f"Plotting ROC curve for {model_name}")
        
        from sklearn.metrics import roc_auc_score
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to: {save_path}")
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   model_name: str = "Model",
                                   save_name: str = "precision_recall_curve.png") -> None:
        """Plot Precision-Recall curve"""
        logger.info(f"Plotting Precision-Recall curve for {model_name}")
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=model_name)
        
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-Recall curve saved to: {save_path}")
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               top_n: int = 15,
                               title: str = "Feature Importance",
                               save_name: str = "feature_importance.png") -> None:
        """Plot feature importance"""
        logger.info(f"Plotting feature importance: {title}")
        
        # Get top N features
        plot_df = importance_df.head(top_n).copy()
        
        # Determine column name for importance values
        importance_col = 'importance' if 'importance' in plot_df.columns else 'abs_coefficient'
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(plot_df)), plot_df[importance_col], color='steelblue')
        plt.yticks(range(len(plot_df)), plot_df['feature'])
        plt.xlabel('Importance', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to: {save_path}")
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                            metrics: List[str] = None,
                            save_name: str = "model_comparison.png") -> None:
        """Plot model comparison"""
        logger.info("Plotting model comparison")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        plot_df = comparison_df[available_metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(plot_df.index))
        width = 0.15
        
        for i, metric in enumerate(available_metrics):
            offset = width * (i - len(available_metrics)/2)
            ax.bar(x + offset, plot_df[metric], width, label=metric.upper())
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df.index, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to: {save_path}")