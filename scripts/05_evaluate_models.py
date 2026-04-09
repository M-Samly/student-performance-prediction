### scripts/05_evaluate_models.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualizer import ModelVisualizer
from src.evaluation.explainability import ModelExplainer
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR
from src.utils.file_handler import load_csv, load_model, load_json, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_single_model(model, model_name: str, X_test: pd.DataFrame, 
                         y_test: pd.Series, X_train: pd.DataFrame = None):
    """Evaluate a single model"""
    
    logger.info(f"\nEvaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_all_metrics(y_test, y_pred, y_proba)
    
    # Get confusion matrix and classification report
    cm = evaluator.get_confusion_matrix(y_test, y_pred)
    report = evaluator.get_classification_report(y_test, y_pred)
    
    logger.info(f"\n{model_name} Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"\nClassification Report:\n{report}")
    
    # Visualizations
    visualizer = ModelVisualizer()
    
    # Confusion matrix
    visualizer.plot_confusion_matrix(
        y_test, y_pred,
        title=f"{model_name} - Confusion Matrix",
        save_name=f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )
    
    # ROC curve
    visualizer.plot_roc_curve(
        y_test, y_proba,
        model_name=model_name,
        save_name=f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
    )
    
    # Precision-Recall curve
    visualizer.plot_precision_recall_curve(
        y_test, y_proba,
        model_name=model_name,
        save_name=f"{model_name.lower().replace(' ', '_')}_pr_curve.png"
    )
    
    # SHAP analysis (if training data provided)
    if X_train is not None:
        try:
            logger.info(f"Performing SHAP analysis for {model_name}...")
            
            # Determine model type for SHAP
            model_type = 'tree' if 'forest' in model_name.lower() or 'boosting' in model_name.lower() else 'linear'
            
            explainer = ModelExplainer(model, X_train)
            explainer.create_shap_explainer(model_type=model_type)
            
            # Use a sample for SHAP (to save time)
            X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
            
            explainer.plot_shap_summary(
                X_sample,
                save_name=f"{model_name.lower().replace(' ', '_')}_shap_summary.png"
            )
            
            explainer.plot_shap_importance(
                X_sample,
                save_name=f"{model_name.lower().replace(' ', '_')}_shap_importance.png"
            )
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed for {model_name}: {str(e)}")
    
    return metrics

def main():
    """Evaluate all trained models"""
    
    logger.info("="*50)
    logger.info("STEP 5: MODEL EVALUATION")
    logger.info("="*50)
    
    # Load test data
    splits_dir = PROCESSED_DATA_DIR / "train_test_split"
    train_data = load_csv(splits_dir / "train.csv")
    test_data = load_csv(splits_dir / "test.csv")
    
    X_train = train_data.drop('pass_fail', axis=1)
    X_test = test_data.drop('pass_fail', axis=1)
    y_test = test_data['pass_fail']
    
    logger.info(f"Test set size: {X_test.shape}")
    
    # Dictionary to store all results
    all_results = {}
    
    # ========================================
    # Evaluate Baseline Model
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("EVALUATING BASELINE MODEL")
    logger.info("="*50)
    
    baseline_path = MODELS_DIR / "baseline" / "logistic_regression.pkl"
    if baseline_path.exists():
        baseline_model = load_model(baseline_path)
        baseline_metrics = evaluate_single_model(
            baseline_model, "Logistic Regression", X_test, y_test, X_train
        )
        all_results['Logistic_Regression'] = baseline_metrics
    else:
        logger.warning("Baseline model not found")
    
    # ========================================
    # Evaluate Random Forest
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("EVALUATING RANDOM FOREST")
    logger.info("="*50)
    
    rf_path = MODELS_DIR / "improved" / "random_forest.pkl"
    if rf_path.exists():
        rf_model = load_model(rf_path)
        rf_metrics = evaluate_single_model(
            rf_model, "Random Forest", X_test, y_test, X_train
        )
        all_results['Random_Forest'] = rf_metrics
        
        # Plot feature importance
        visualizer = ModelVisualizer()
        rf_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        visualizer.plot_feature_importance(
            rf_importance,
            title="Random Forest - Feature Importance",
            save_name="random_forest_feature_importance.png"
        )
    else:
        logger.warning("Random Forest model not found")
    
    # ========================================
    # Evaluate Gradient Boosting
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("EVALUATING GRADIENT BOOSTING")
    logger.info("="*50)
    
    gb_path = MODELS_DIR / "improved" / "gradient_boosting.pkl"
    if gb_path.exists():
        gb_model = load_model(gb_path)
        gb_metrics = evaluate_single_model(
            gb_model, "Gradient Boosting", X_test, y_test, X_train
        )
        all_results['Gradient_Boosting'] = gb_metrics
        
        # Plot feature importance
        visualizer = ModelVisualizer()
        gb_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        visualizer.plot_feature_importance(
            gb_importance,
            title="Gradient Boosting - Feature Importance",
            save_name="gradient_boosting_feature_importance.png"
        )
    else:
        logger.warning("Gradient Boosting model not found")
    
    # ========================================
    # Model Comparison
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("MODEL COMPARISON")
    logger.info("="*50)
    
    comparison_df = pd.DataFrame(all_results).T
    logger.info("\nFinal Model Comparison:")
    logger.info("\n" + comparison_df.to_string())
    
    # Save comparison
    comparison_path = REPORTS_DIR / "final_model_comparison.csv"
    comparison_df.to_csv(comparison_path)
    logger.info(f"\nComparison saved to: {comparison_path}")
    
    # Plot comparison
    visualizer = ModelVisualizer()
    visualizer.plot_model_comparison(
        comparison_df,
        save_name="final_model_comparison.png"
    )
    
    # Determine best model
    best_model_name = comparison_df['f1_score'].idxmax()
    best_metrics = comparison_df.loc[best_model_name].to_dict()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"BEST MODEL: {best_model_name}")
    logger.info(f"{'='*50}")
    logger.info(f"F1-Score: {best_metrics['f1_score']:.4f}")
    logger.info(f"Accuracy: {best_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {best_metrics['precision']:.4f}")
    logger.info(f"Recall: {best_metrics['recall']:.4f}")
    logger.info(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    # Save best model info
    best_model_info = {
        'best_model': best_model_name,
        'metrics': best_metrics,
        'all_results': comparison_df.to_dict()
    }
    
    best_model_path = REPORTS_DIR / "best_model_evaluation.json"
    save_json(best_model_info, best_model_path)
    
    logger.info(f"\nBest model info saved to: {best_model_path}")
    logger.info(f"\nAll figures saved to: {FIGURES_DIR}")
    
    logger.info("\n" + "="*50)
    logger.info("MODEL EVALUATION COMPLETE")
    logger.info("="*50)

if __name__ == "__main__":
    main()