### scripts/04_train_improved_models.py
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.improved_models import RandomForestModel, GradientBoostingModel
from src.models.trainer import ModelTrainer
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED
from src.utils.file_handler import load_csv, save_model, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Train improved models (Random Forest and Gradient Boosting)"""
    
    logger.info("="*50)
    logger.info("STEP 4: TRAINING IMPROVED MODELS")
    logger.info("="*50)
    
    # Load data splits
    splits_dir = PROCESSED_DATA_DIR / "train_test_split"
    
    train_data = load_csv(splits_dir / "train.csv")
    val_data = load_csv(splits_dir / "validation.csv")
    test_data = load_csv(splits_dir / "test.csv")
    
    # Separate features and target
    X_train = train_data.drop('pass_fail', axis=1)
    y_train = train_data['pass_fail']
    
    X_val = val_data.drop('pass_fail', axis=1)
    y_val = val_data['pass_fail']
    
    X_test = test_data.drop('pass_fail', axis=1)
    y_test = test_data['pass_fail']
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Create improved models directory
    improved_dir = MODELS_DIR / "improved"
    improved_dir.mkdir(exist_ok=True, parents=True)
    
    # Dictionary to store results
    all_results = {}
    
    # ========================================
    # Train Random Forest
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("TRAINING RANDOM FOREST")
    logger.info("="*50)
    
    rf_model = RandomForestModel()
    rf_trainer = ModelTrainer(rf_model, "Random_Forest")
    rf_history = rf_trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    logger.info("\nEvaluating Random Forest on test set...")
    rf_test_metrics = rf_model.evaluate(X_test, y_test)
    
    # Get feature importance
    rf_importance = rf_model.get_feature_importance()
    logger.info("\nTop 10 Most Important Features (Random Forest):")
    logger.info(rf_importance.head(10).to_string())
    
    # Save Random Forest model
    rf_path = improved_dir / "random_forest.pkl"
    save_model(rf_model.get_model(), rf_path)
    
    rf_metadata = {
        'model_name': 'Random Forest',
        'random_seed': RANDOM_SEED,
        'training_history': rf_history,
        'test_metrics': rf_test_metrics,
        'n_features': X_train.shape[1],
        'feature_names': X_train.columns.tolist()
    }
    
    rf_metadata_path = improved_dir / "random_forest_metadata.json"
    save_json(rf_metadata, rf_metadata_path)
    
    rf_importance_path = improved_dir / "random_forest_feature_importance.csv"
    rf_importance.to_csv(rf_importance_path, index=False)
    
    all_results['random_forest'] = rf_test_metrics
    
    logger.info(f"\nRandom Forest saved to: {rf_path}")
    
    # ========================================
    # Train Gradient Boosting
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("TRAINING GRADIENT BOOSTING")
    logger.info("="*50)
    
    gb_model = GradientBoostingModel()
    gb_trainer = ModelTrainer(gb_model, "Gradient_Boosting")
    gb_history = gb_trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    logger.info("\nEvaluating Gradient Boosting on test set...")
    gb_test_metrics = gb_model.evaluate(X_test, y_test)
    
    # Get feature importance
    gb_importance = gb_model.get_feature_importance()
    logger.info("\nTop 10 Most Important Features (Gradient Boosting):")
    logger.info(gb_importance.head(10).to_string())
    
    # Save Gradient Boosting model
    gb_path = improved_dir / "gradient_boosting.pkl"
    save_model(gb_model.get_model(), gb_path)
    
    gb_metadata = {
        'model_name': 'Gradient Boosting',
        'random_seed': RANDOM_SEED,
        'training_history': gb_history,
        'test_metrics': gb_test_metrics,
        'n_features': X_train.shape[1],
        'feature_names': X_train.columns.tolist()
    }
    
    gb_metadata_path = improved_dir / "gradient_boosting_metadata.json"
    save_json(gb_metadata, gb_metadata_path)
    
    gb_importance_path = improved_dir / "gradient_boosting_feature_importance.csv"
    gb_importance.to_csv(gb_importance_path, index=False)
    
    all_results['gradient_boosting'] = gb_test_metrics
    
    logger.info(f"\nGradient Boosting saved to: {gb_path}")
    
    # ========================================
    # Compare all models
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("MODEL COMPARISON")
    logger.info("="*50)
    
    comparison_df = pd.DataFrame(all_results).T
    logger.info("\n" + comparison_df.to_string())
    
    # Save comparison
    comparison_path = improved_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path)
    
    # Determine best model
    best_model_name = comparison_df['f1'].idxmax()
    best_f1 = comparison_df.loc[best_model_name, 'f1']
    
    logger.info(f"\nBest model based on F1-score: {best_model_name} (F1: {best_f1:.4f})")
    
    # Mark best model
    best_model_info = {
        'best_model': best_model_name,
        'best_f1_score': float(best_f1),
        'all_metrics': comparison_df.to_dict()
    }
    
    best_model_path = improved_dir / "best_model_info.json"
    save_json(best_model_info, best_model_path)
    
    logger.info(f"\nComparison saved to: {comparison_path}")
    logger.info(f"Best model info saved to: {best_model_path}")
    
    logger.info("\n" + "="*50)
    logger.info("IMPROVED MODELS TRAINING COMPLETE")
    logger.info("="*50)

if __name__ == "__main__":
    main()