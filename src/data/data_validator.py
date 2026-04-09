###src/data/data_validator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate data quality and integrity for OULAD dataset"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validation_report = {}
    
    def check_missing_values(self) -> Dict:
        """Check for missing values in dataset"""
        logger.info("Checking for missing values...")
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_report = pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct
        })
        
        missing_report = missing_report[missing_report['missing_count'] > 0]
        missing_report = missing_report.sort_values('missing_percentage', ascending=False)
        
        self.validation_report['missing_values'] = missing_report.to_dict()
        
        logger.info(f"Found {len(missing_report)} columns with missing values")
        
        return missing_report.to_dict()
    
    def check_duplicates(self, subset: List[str] = None) -> int:
        """
        Check for duplicate rows
        
        Args:
            subset: List of columns to check for duplicates
        
        Returns:
            Number of duplicate rows
        """
        logger.info("Checking for duplicate rows...")
        
        duplicates = self.df.duplicated(subset=subset).sum()
        
        self.validation_report['duplicates'] = int(duplicates)
        
        logger.info(f"Found {duplicates} duplicate rows")
        
        return int(duplicates)
    
    def check_target_distribution(self, target_col: str = 'final_result') -> Dict:
        """Check distribution of target variable"""
        logger.info("Checking target distribution...")
        
        if target_col not in self.df.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return {}
        
        distribution = self.df[target_col].value_counts()
        distribution_pct = (distribution / len(self.df) * 100).round(2)
        
        target_report = {
            'counts': distribution.to_dict(),
            'percentages': distribution_pct.to_dict()
        }
        
        self.validation_report['target_distribution'] = target_report
        
        logger.info(f"Target distribution: {distribution.to_dict()}")
        
        return target_report
    
    def check_data_types(self) -> Dict:
        """Check data types of all columns"""
        logger.info("Checking data types...")
        
        dtype_report = {
            'numeric': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical': list(self.df.select_dtypes(include=['object']).columns),
            'boolean': list(self.df.select_dtypes(include=['bool']).columns)
        }
        
        self.validation_report['data_types'] = dtype_report
        
        logger.info(f"Numeric columns: {len(dtype_report['numeric'])}")
        logger.info(f"Categorical columns: {len(dtype_report['categorical'])}")
        
        return dtype_report
    
    def check_score_ranges(self) -> Dict:
        """Check if scores are in valid ranges"""
        logger.info("Checking score ranges...")
        
        score_cols = [col for col in self.df.columns if 'score' in col.lower()]
        
        range_report = {}
        for col in score_cols:
            if col in self.df.columns:
                col_data = self.df[col].dropna()
                range_report[col] = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'out_of_range': int(((col_data < 0) | (col_data > 100)).sum())
                }
        
        self.validation_report['score_ranges'] = range_report
        
        return range_report
    
    def get_validation_summary(self) -> Dict:
        """Get complete validation report"""
        return self.validation_report
    
    def validate_all(self, target_col: str = 'final_result') -> Dict:
        """
        Run all validation checks
        
        Args:
            target_col: Target column name
        
        Returns:
            Complete validation report
        """
        logger.info("Running all validation checks...")
        
        self.check_missing_values()
        self.check_duplicates()
        self.check_target_distribution(target_col)
        self.check_data_types()
        self.check_score_ranges()
        
        # Additional summary
        self.validation_report['total_rows'] = int(len(self.df))
        self.validation_report['total_columns'] = int(len(self.df.columns))
        
        logger.info("Validation complete")
        
        return self.validation_report