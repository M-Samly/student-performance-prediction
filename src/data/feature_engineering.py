###src/data/feature_engineering.py
import pandas as pd
import numpy as np
from typing import List, Dict
from src.config import PASS_CATEGORIES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Create new features from OULAD data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.new_features = []
    
    def create_engagement_features(self) -> pd.DataFrame:
        """Create engagement-related features from VLE data"""
        logger.info("Creating engagement features...")
        
        # Engagement intensity (clicks per session)
        if 'total_clicks' in self.df.columns and 'total_sessions' in self.df.columns:
            self.df['engagement_intensity'] = np.where(
                self.df['total_sessions'] > 0,
                self.df['total_clicks'] / self.df['total_sessions'],
                0
            )
            self.new_features.append('engagement_intensity')
        
        # Activity span (days between first and last activity)
        if 'first_activity_date' in self.df.columns and 'last_activity_date' in self.df.columns:
            self.df['activity_span'] = self.df['last_activity_date'] - self.df['first_activity_date']
            self.new_features.append('activity_span')
        
        # Early engagement indicator
        if 'first_activity_date' in self.df.columns:
            self.df['early_engagement'] = (self.df['first_activity_date'] <= 0).astype(int)
            self.new_features.append('early_engagement')
        
        # High engagement indicator
        if 'total_clicks' in self.df.columns:
            click_median = self.df['total_clicks'].median()
            self.df['high_engagement'] = (self.df['total_clicks'] > click_median).astype(int)
            self.new_features.append('high_engagement')
        
        # No engagement indicator
        if 'total_clicks' in self.df.columns:
            self.df['no_engagement'] = (self.df['total_clicks'] == 0).astype(int)
            self.new_features.append('no_engagement')
        
        logger.info(f"Created engagement features: {self.new_features[-5:]}")
        
        return self.df
    
    def create_assessment_features(self) -> pd.DataFrame:
        """Create assessment-related features"""
        logger.info("Creating assessment features...")
        
        # Score range
        if 'max_score' in self.df.columns and 'min_score' in self.df.columns:
            self.df['score_range'] = self.df['max_score'] - self.df['min_score']
            self.new_features.append('score_range')
        
        # Passing score indicator (avg >= 50)
        if 'avg_score' in self.df.columns:
            self.df['passing_avg_score'] = (self.df['avg_score'] >= 50).astype(int)
            self.new_features.append('passing_avg_score')
        
        # Low score indicator (avg < 40)
        if 'avg_score' in self.df.columns:
            self.df['low_avg_score'] = (self.df['avg_score'] < 40).astype(int)
            self.new_features.append('low_avg_score')
        
        # Assessment participation indicator
        if 'num_assessments' in self.df.columns:
            self.df['has_assessments'] = (self.df['num_assessments'] > 0).astype(int)
            self.new_features.append('has_assessments')
        
        # Score consistency (inverse of std)
        if 'std_score' in self.df.columns:
            self.df['score_consistency'] = np.where(
                self.df['std_score'] > 0,
                1 / (1 + self.df['std_score']),
                1
            )
            self.new_features.append('score_consistency')
        
        logger.info(f"Created assessment features")
        
        return self.df
    
    def create_registration_features(self) -> pd.DataFrame:
        """Create registration-related features"""
        logger.info("Creating registration features...")
        
        # Convert date_registration to numeric (handle '?' values)
        if 'date_registration' in self.df.columns:
            self.df['date_registration'] = pd.to_numeric(
                self.df['date_registration'], 
                errors='coerce'
            ).fillna(0)
            
            # Early registration indicator (before course start)
            self.df['early_registration'] = (self.df['date_registration'] < 0).astype(int)
            self.new_features.append('early_registration')
            
            # Late registration indicator
            self.df['late_registration'] = (self.df['date_registration'] > 0).astype(int)
            self.new_features.append('late_registration')
        
        logger.info(f"Created registration features")
        
        return self.df
    
    def create_demographic_features(self) -> pd.DataFrame:
        """Create demographic-related features"""
        logger.info("Creating demographic features...")
        
        # Previous attempt indicator
        if 'num_of_prev_attempts' in self.df.columns:
            self.df['is_repeater'] = (self.df['num_of_prev_attempts'] > 0).astype(int)
            self.new_features.append('is_repeater')
        
        # High credit load indicator
        if 'studied_credits' in self.df.columns:
            credit_median = self.df['studied_credits'].median()
            self.df['high_credit_load'] = (self.df['studied_credits'] > credit_median).astype(int)
            self.new_features.append('high_credit_load')
        
        logger.info(f"Created demographic features")
        
        return self.df
    
    def create_risk_indicators(self) -> pd.DataFrame:
        """Create risk indicator features"""
        logger.info("Creating risk indicators...")
        
        risk_factors = []
        
        # Risk: No VLE engagement
        if 'no_engagement' in self.df.columns:
            risk_factors.append('no_engagement')
        
        # Risk: Low assessment score
        if 'low_avg_score' in self.df.columns:
            risk_factors.append('low_avg_score')
        
        # Risk: Late registration
        if 'late_registration' in self.df.columns:
            risk_factors.append('late_registration')
        
        # Risk: Is repeater
        if 'is_repeater' in self.df.columns:
            risk_factors.append('is_repeater')
        
        # Risk: Unregistered
        if 'is_unregistered' in self.df.columns:
            risk_factors.append('is_unregistered')
        
        # Combined risk score
        if risk_factors:
            self.df['combined_risk_score'] = self.df[risk_factors].sum(axis=1)
            self.new_features.append('combined_risk_score')
        
        # Risk category
        if 'combined_risk_score' in self.df.columns:
            self.df['risk_category'] = pd.cut(
                self.df['combined_risk_score'],
                bins=[-1, 1, 2, 10],
                labels=['Low', 'Medium', 'High']
            )
            self.new_features.append('risk_category')
        
        logger.info(f"Created risk indicators")
        
        return self.df
    
    def create_all_features(self) -> pd.DataFrame:
        """Create all features at once"""
        logger.info("Creating all features...")
        
        self.create_engagement_features()
        self.create_assessment_features()
        self.create_registration_features()
        self.create_demographic_features()
        self.create_risk_indicators()
        
        logger.info(f"Total new features created: {len(self.new_features)}")
        logger.info(f"New features: {self.new_features}")
        
        return self.df
    
    def get_feature_list(self) -> List[str]:
        """Get list of all created features"""
        return self.new_features
    
    def get_engineered_data(self) -> pd.DataFrame:
        """Get DataFrame with engineered features"""
        return self.df