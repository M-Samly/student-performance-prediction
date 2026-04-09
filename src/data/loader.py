###src/data/loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from src.config import RAW_DATA_DIR, CSV_FILES, STUDENT_ID_COL, MODULE_COL, PRESENTATION_COL
from src.utils.logger import get_logger
from src.utils.file_handler import load_csv

logger = get_logger(__name__)


class DataLoader:
    """Load and merge all OULAD dataset files"""
    
    def __init__(self):
        self.student_info = None
        self.student_assessment = None
        self.student_registration = None
        self.student_vle = None
        self.assessments = None
        self.courses = None
        self.vle = None
        self.merged_data = None
    
    def load_all_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from raw data directory
        
        Returns:
            Dictionary with all loaded DataFrames
        """
        logger.info("Loading all OULAD data files...")
        
        try:
            # Load all files
            self.student_info = load_csv(RAW_DATA_DIR / CSV_FILES['student_info'])
            self.student_assessment = load_csv(RAW_DATA_DIR / CSV_FILES['student_assessment'])
            self.student_registration = load_csv(RAW_DATA_DIR / CSV_FILES['student_registration'])
            self.student_vle = load_csv(RAW_DATA_DIR / CSV_FILES['student_vle'])
            self.assessments = load_csv(RAW_DATA_DIR / CSV_FILES['assessments'])
            self.courses = load_csv(RAW_DATA_DIR / CSV_FILES['courses'])
            self.vle = load_csv(RAW_DATA_DIR / CSV_FILES['vle'])
            
            logger.info("All files loaded successfully")
            
            return {
                'student_info': self.student_info,
                'student_assessment': self.student_assessment,
                'student_registration': self.student_registration,
                'student_vle': self.student_vle,
                'assessments': self.assessments,
                'courses': self.courses,
                'vle': self.vle
            }
        except Exception as e:
            logger.error(f"Error loading data files: {str(e)}")
            raise
    
    def _aggregate_student_vle(self) -> pd.DataFrame:
        """Aggregate VLE interactions per student-module-presentation"""
        logger.info("Aggregating student VLE interactions...")
        
        try:
            vle_agg = self.student_vle.groupby(
                [MODULE_COL, PRESENTATION_COL, STUDENT_ID_COL]
            ).agg({
                'sum_click': ['sum', 'mean', 'count'],
                'date': ['min', 'max']
            }).reset_index()
            
            # Flatten column names
            vle_agg.columns = [
                MODULE_COL, PRESENTATION_COL, STUDENT_ID_COL,
                'total_clicks', 'avg_clicks_per_session', 'total_sessions',
                'first_activity_date', 'last_activity_date'
            ]
            
            logger.info(f"VLE aggregation complete: {len(vle_agg)} records")
            
            return vle_agg
        except Exception as e:
            logger.error(f"Error aggregating VLE data: {e}")
            raise
    
    def _aggregate_student_assessments(self) -> pd.DataFrame:
        """Aggregate assessment scores per student-module-presentation"""
        logger.info("Aggregating student assessment scores...")
        
        try:
            # Verify required columns exist
            required_cols = ['id_assessment', STUDENT_ID_COL, 'score', 'date_submitted', 'is_banked']
            missing_cols = [col for col in required_cols if col not in self.student_assessment.columns]
            
            if missing_cols:
                logger.error(f"Missing columns in student_assessment: {missing_cols}")
                raise ValueError(f"Required columns missing: {missing_cols}")
            
            # Check assessments has required columns
            if 'id_assessment' not in self.assessments.columns:
                logger.error("assessments.csv missing 'id_assessment' column")
                raise ValueError("assessments.csv must have 'id_assessment' column")
            
            # Clean the student_assessment data
            student_assessment_clean = self.student_assessment.copy()
            
            # Convert score to numeric (handle '?' and other non-numeric values)
            student_assessment_clean['score'] = pd.to_numeric(
                student_assessment_clean['score'], 
                errors='coerce'
            )
            
            # Convert date_submitted to numeric
            student_assessment_clean['date_submitted'] = pd.to_numeric(
                student_assessment_clean['date_submitted'], 
                errors='coerce'
            )
            
            # Drop rows where score is NaN (were non-numeric)
            initial_rows = len(student_assessment_clean)
            student_assessment_clean = student_assessment_clean.dropna(subset=['score'])
            dropped_rows = initial_rows - len(student_assessment_clean)
            
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} assessment records with invalid scores")
            
            # Merge assessments info with student assessments
            assessment_merged = student_assessment_clean.merge(
                self.assessments,
                on='id_assessment',
                how='left'
            )
            
            logger.info(f"Merged assessment data: {len(assessment_merged)} records")
            
            # Check if we have module and presentation columns
            if MODULE_COL not in assessment_merged.columns:
                logger.error(f"Column '{MODULE_COL}' not found after merge")
                logger.error(f"Available columns: {assessment_merged.columns.tolist()}")
                raise ValueError(f"Column '{MODULE_COL}' missing after merge")
            
            if PRESENTATION_COL not in assessment_merged.columns:
                logger.error(f"Column '{PRESENTATION_COL}' not found after merge")
                logger.error(f"Available columns: {assessment_merged.columns.tolist()}")
                raise ValueError(f"Column '{PRESENTATION_COL}' missing after merge")
            
            # Aggregate per student-module-presentation
            assessment_agg = assessment_merged.groupby(
                [MODULE_COL, PRESENTATION_COL, STUDENT_ID_COL]
            ).agg({
                'score': ['mean', 'std', 'min', 'max', 'count'],
                'date_submitted': ['mean'],
                'is_banked': ['sum']
            }).reset_index()
            
            # Flatten column names
            assessment_agg.columns = [
                MODULE_COL, PRESENTATION_COL, STUDENT_ID_COL,
                'avg_score', 'std_score', 'min_score', 'max_score', 'num_assessments',
                'avg_submission_date', 'num_banked'
            ]
            
            # Fill NaN std with 0 (for students with single assessment)
            assessment_agg['std_score'] = assessment_agg['std_score'].fillna(0)
            
            logger.info(f"Assessment aggregation complete: {len(assessment_agg)} records")
            
            return assessment_agg
            
        except Exception as e:
            logger.error(f"Error aggregating assessments: {e}")
            logger.error(f"student_assessment columns: {self.student_assessment.columns.tolist()}")
            logger.error(f"assessments columns: {self.assessments.columns.tolist()}")
            raise
        
    def _process_registration(self) -> pd.DataFrame:
        """Process registration data"""
        logger.info("Processing registration data...")
        
        reg_data = self.student_registration.copy()
        
        # Convert ? to NaN
        reg_data['date_unregistration'] = reg_data['date_unregistration'].replace('?', np.nan)
        reg_data['date_unregistration'] = pd.to_numeric(reg_data['date_unregistration'], errors='coerce')
        
        # Create unregistered flag
        reg_data['is_unregistered'] = reg_data['date_unregistration'].notna().astype(int)
        
        logger.info(f"Registration processing complete: {len(reg_data)} records")
        
        return reg_data
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge all datasets into a single DataFrame
        
        Returns:
            Merged DataFrame with all student data
        """
        logger.info("Merging all datasets...")
        
        if self.student_info is None:
            raise ValueError("Data not loaded. Call load_all_files() first.")
        
        try:
            # Start with student info as base
            merged = self.student_info.copy()
            logger.info(f"Base student info: {len(merged)} rows")
            
            # Merge with courses info
            merged = merged.merge(
                self.courses,
                on=[MODULE_COL, PRESENTATION_COL],
                how='left'
            )
            logger.info(f"After courses merge: {len(merged)} rows")
            
            # Process and merge registration
            reg_data = self._process_registration()
            merged = merged.merge(
                reg_data,
                on=[MODULE_COL, PRESENTATION_COL, STUDENT_ID_COL],
                how='left'
            )
            logger.info(f"After registration merge: {len(merged)} rows")
            
            # Aggregate and merge VLE data
            vle_agg = self._aggregate_student_vle()
            merged = merged.merge(
                vle_agg,
                on=[MODULE_COL, PRESENTATION_COL, STUDENT_ID_COL],
                how='left'
            )
            logger.info(f"After VLE merge: {len(merged)} rows")
            
            # Aggregate and merge assessment data
            assessment_agg = self._aggregate_student_assessments()
            merged = merged.merge(
                assessment_agg,
                on=[MODULE_COL, PRESENTATION_COL, STUDENT_ID_COL],
                how='left'
            )
            logger.info(f"After assessment merge: {len(merged)} rows")
            
            self.merged_data = merged
            logger.info(f"Final merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            raise
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of loaded data
        
        Returns:
            Dictionary with summary information
        """
        if self.merged_data is None:
            raise ValueError("Data not merged. Call merge_datasets() first.")
        
        summary = {
            'total_rows': int(len(self.merged_data)),
            'total_columns': int(len(self.merged_data.columns)),
            'column_names': list(self.merged_data.columns),
            'missing_values': self.merged_data.isnull().sum().to_dict(),
            'data_types': self.merged_data.dtypes.astype(str).to_dict(),
            'unique_students': int(self.merged_data[STUDENT_ID_COL].nunique()),
            'unique_modules': int(self.merged_data[MODULE_COL].nunique()),
            'target_distribution': self.merged_data['final_result'].value_counts().to_dict()
        }
        
        return summary