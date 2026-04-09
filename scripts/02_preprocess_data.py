import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.config import PROCESSED_DATA_DIR, RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE
from src.utils.file_handler import load_csv, save_csv, save_json, save_model
from src.utils.logger import get_logger
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)


def main():
    """Preprocess data and create features"""
    
    logger.info("="*60)
    logger.info("STEP 2: DATA PREPROCESSING AND FEATURE ENGINEERING")
    logger.info("="*60)
    
    # Load merged data
    merged_data_path = PROCESSED_DATA_DIR / "merged_raw_data.csv"
    df = load_csv(merged_data_path)
    
    logger.info(f"Loaded data: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(df)
    
    # Handle missing values
    logger.info("\nHandling missing values...")
    df_clean = preprocessor.handle_missing_values()
    
    # Remove duplicates
    df_clean = preprocessor.remove_duplicates()
    
    # Create target variable
    logger.info("\nCreating target variable...")
    df_clean = preprocessor.create_target_variable()
    
    # Save cleaned data
    clean_data_path = PROCESSED_DATA_DIR / "cleaned_data.csv"
    save_csv(df_clean, clean_data_path)
    logger.info(f"Cleaned data saved to: {clean_data_path}")
    
    # Feature Engineering
    logger.info("\n" + "="*60)
    logger.info("FEATURE ENGINEERING")
    logger.info("="*60)
    
    engineer = FeatureEngineer(df_clean)
    df_features = engineer.create_all_features()
    
    # Encode categorical variables
    logger.info("\nEncoding categorical variables...")
    preprocessor = DataPreprocessor(df_features)
    df_features = preprocessor.encode_categorical_variables()
    
    # Save feature-engineered data
    features_data_path = PROCESSED_DATA_DIR / "featured_data.csv"
    save_csv(df_features, features_data_path)
    logger.info(f"Featured data saved to: {features_data_path}")
    
    # Prepare features and target
    logger.info("\n" + "="*60)
    logger.info("PREPARING FEATURES AND TARGET")
    logger.info("="*60)
    
    X, y = preprocessor.prepare_features_target()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Split data: train, validation, test
    logger.info("\nSplitting data...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # Second split: separate validation from train
    val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    logger.info(f"\nData split:")
    logger.info(f"  Training: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    logger.info(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    logger.info(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Class distribution
    logger.info(f"\nClass distribution:")
    logger.info(f"  Training - Pass: {y_train.sum()}, Fail: {len(y_train)-y_train.sum()}")
    logger.info(f"  Validation - Pass: {y_val.sum()}, Fail: {len(y_val)-y_val.sum()}")
    logger.info(f"  Test - Pass: {y_test.sum()}, Fail: {len(y_test)-y_test.sum()}")
    
    # Save splits
    splits_dir = PROCESSED_DATA_DIR / "train_test_split"
    splits_dir.mkdir(exist_ok=True)
    
    # Combine X and y for saving
    train_data = X_train.copy()
    train_data['pass_fail'] = y_train.values
    save_csv(train_data, splits_dir / "train.csv")
    
    val_data = X_val.copy()
    val_data['pass_fail'] = y_val.values
    save_csv(val_data, splits_dir / "validation.csv")
    
    test_data = X_test.copy()
    test_data['pass_fail'] = y_test.values
    save_csv(test_data, splits_dir / "test.csv")
    
    # Save preprocessing info
    preprocessing_info = {
        'feature_columns': list(X.columns),
        'target_column': 'pass_fail',
        'n_features': len(X.columns),
        'n_samples': len(df_features),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'random_seed': RANDOM_SEED,
        'new_features_created': engineer.get_feature_list()
    }
    
    save_json(preprocessing_info, PROCESSED_DATA_DIR / "preprocessing_info.json")
    
    # Save scaler
    save_model(preprocessor.scaler, PROCESSED_DATA_DIR / "scaler.pkl")
    
    logger.info(f"\nAll splits saved to: {splits_dir}")
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    main()