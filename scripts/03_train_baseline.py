### scripts/03_train_baseline.py
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.baseline import BaselineModel
from src.models.trainer import ModelTrainer
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED
from src.utils.file_handler import load_csv, save_model, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Train baseline logistic regression model"""
    
    logger.info("="*50)
    logger.info("STEP 3: TRAINING BASELINE MODEL")
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
    
    # Initialize baseline model
    baseline = BaselineModel()
    
    # Train model using trainer
    trainer = ModelTrainer(baseline, "Logistic_Regression_Baseline")
    training_history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = baseline.evaluate(X_test, y_test)
    
    # Get feature importance
    feature_importance = baseline.get_feature_importance()
    logger.info("\nTop 10 Most Important Features:")
    logger.info(feature_importance.head(10).to_string())
    
    # Save model
    baseline_dir = MODELS_DIR / "baseline"
    baseline_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = baseline_dir / "logistic_regression.pkl"
    save_model(baseline.get_model(), model_path)
    
    # Save metadata
    metadata = {
        'model_name': 'Logistic Regression Baseline',
        'random_seed': RANDOM_SEED,
        'training_history': training_history,
        'test_metrics': test_metrics,
        'n_features': X_train.shape[1],
        'feature_names': X_train.columns.tolist()
    }
    
    metadata_path = baseline_dir / "model_metadata.json"
    save_json(metadata, metadata_path)
    
    # Save feature importance
    feature_importance_path = baseline_dir / "feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    
    logger.info(f"\nModel saved to: {model_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"Feature importance saved to: {feature_importance_path}")
    
    logger.info("\n" + "="*50)
    logger.info("BASELINE MODEL TRAINING COMPLETE")
    logger.info("="*50)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1-Score: {test_metrics['f1']:.4f}")
    logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()