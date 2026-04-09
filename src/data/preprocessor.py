###src/data/preprocessor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.config import (
    RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE, 
    PASS_CATEGORIES, FAIL_CATEGORIES, TARGET_COL
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocess and clean OULAD student data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = None
        
    def handle_missing_values(self, strategy: Dict = None) -> pd.DataFrame:
        """
        Handle missing values in dataset
        
        Args:
            strategy: Dictionary with column names and strategies
        
        Returns:
            DataFrame with handled missing values
        """
        logger.info("Handling missing values...")
        
        initial_missing = self.df.isnull().sum().sum()
        logger.info(f"Initial missing values: {initial_missing}")
        
        # Handle specific columns
        # VLE columns - fill with 0 (no activity)
        vle_cols = ['total_clicks', 'avg_clicks_per_session', 'total_sessions',
                    'first_activity_date', 'last_activity_date']
        for col in vle_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)
        
        # Assessment columns - fill with appropriate values
        assessment_cols = ['avg_score', 'std_score', 'min_score', 'max_score', 
                          'num_assessments', 'avg_submission_date', 'num_banked']
        for col in assessment_cols:
            if col in self.df.columns:
                if 'score' in col:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(0)
        
        # Registration columns
        if 'date_registration' in self.df.columns:
            self.df['date_registration'] = self.df['date_registration'].fillna(0)
        if 'date_unregistration' in self.df.columns:
            self.df['date_unregistration'] = self.df['date_unregistration'].fillna(-999)
        if 'is_unregistered' in self.df.columns:
            self.df['is_unregistered'] = self.df['is_unregistered'].fillna(0)
        
        # IMD band - fill with most common
        if 'imd_band' in self.df.columns:
            self.df['imd_band'] = self.df['imd_band'].fillna(self.df['imd_band'].mode()[0])
        
        # Numeric columns - fill with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                logger.info(f"Filled {col} with median: {median_val}")
        
        # Categorical columns - fill with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().any():
                if not self.df[col].mode().empty:
                    mode_val = self.df[col].mode()[0]
                else:
                    mode_val = 'Unknown'
                self.df[col] = self.df[col].fillna(mode_val)
                logger.info(f"Filled {col} with mode: {mode_val}")
        
        final_missing = self.df.isnull().sum().sum()
        logger.info(f"Final missing values: {final_missing}")
        
        return self.df
    
    def remove_duplicates(self, subset: List[str] = None) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            subset: Columns to consider for identifying duplicates
        
        Returns:
            DataFrame without duplicates
        """
        logger.info("Removing duplicates...")
        
        initial_rows = len(self.df)
        self.df.drop_duplicates(subset=subset, inplace=True)
        final_rows = len(self.df)
        
        removed = initial_rows - final_rows
        logger.info(f"Removed {removed} duplicate rows")
        
        return self.df
    
    def create_target_variable(self) -> pd.DataFrame:
        """
        Create binary target variable (pass/fail) from final_result
        
        Returns:
            DataFrame with binary target variable
        """
        logger.info("Creating target variable...")
        
        if TARGET_COL not in self.df.columns:
            raise ValueError(f"Column {TARGET_COL} not found in dataset")
        
        # Create binary target: 1 = Pass/Distinction, 0 = Fail/Withdrawn
        self.df['pass_fail'] = self.df[TARGET_COL].apply(
            lambda x: 1 if x in PASS_CATEGORIES else 0
        )
        self.target_column = 'pass_fail'
        
        pass_count = self.df['pass_fail'].sum()
        fail_count = len(self.df) - pass_count
        
        logger.info(f"Target variable created:")
        logger.info(f"  Pass (1): {pass_count} ({pass_count/len(self.df)*100:.1f}%)")
        logger.info(f"  Fail (0): {fail_count} ({fail_count/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def encode_categorical_variables(self, columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            columns: List of columns to encode (None = auto-detect)
        
        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info("Encoding categorical variables...")
        
        # Columns to encode
        if columns is None:
            columns = ['gender', 'region', 'highest_education', 'imd_band', 
                      'age_band', 'disability', 'code_module', 'code_presentation']
        
        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded {col}: {len(le.classes_)} unique values")
        
        return self.df
    
    def normalize_features(self, columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize numeric features using StandardScaler
        
        Args:
            columns: List of columns to normalize (None = all numeric)
        
        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing features...")
        
        if columns is None:
            # Select numeric columns to normalize
            columns = ['total_clicks', 'avg_clicks_per_session', 'total_sessions',
                      'avg_score', 'std_score', 'min_score', 'max_score',
                      'num_assessments', 'studied_credits', 'num_of_prev_attempts',
                      'module_presentation_length', 'date_registration']
            columns = [col for col in columns if col in self.df.columns]
        
        if columns:
            self.df[columns] = self.scaler.fit_transform(self.df[columns])
            logger.info(f"Normalized {len(columns)} features")
        
        return self.df
    
    def prepare_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target vector
        
        Returns:
            Tuple of (X, y)
        """
        logger.info("Preparing features and target...")
        
        # Define feature columns
        feature_cols = [
            # Encoded categorical features
            'gender_encoded', 'region_encoded', 'highest_education_encoded',
            'imd_band_encoded', 'age_band_encoded', 'disability_encoded',
            'code_module_encoded', 'code_presentation_encoded',
            
            # Numeric features
            'num_of_prev_attempts', 'studied_credits', 'module_presentation_length',
            'total_clicks', 'avg_clicks_per_session', 'total_sessions',
            'avg_score', 'std_score', 'min_score', 'max_score', 'num_assessments',
            'date_registration', 'is_unregistered'
        ]
        
        # Filter to only available columns
        available_features = [col for col in feature_cols if col in self.df.columns]
        
        logger.info(f"Available features: {len(available_features)}")
        logger.info(f"Features: {available_features}")
        
        X = self.df[available_features].copy()
        y = self.df['pass_fail'].copy()
        
        self.feature_columns = available_features
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        return X, y
    
    def get_preprocessed_data(self) -> pd.DataFrame:
        """Get the preprocessed DataFrame"""
        return self.df