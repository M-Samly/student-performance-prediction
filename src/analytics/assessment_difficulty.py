### src/analytics/assessment_difficulty.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from pathlib import Path
from src.config import FIGURES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AssessmentDifficulty:
    """Analyze assessment difficulty and discrimination"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def calculate_difficulty_index(self, assessment_columns: List[str]) -> pd.DataFrame:
        """
        Calculate difficulty index for each assessment
        Difficulty = (Mean Score / Max Score) * 100
        
        Args:
            assessment_columns: List of assessment columns
            
        Returns:
            DataFrame with difficulty indices
        """
        logger.info("Calculating assessment difficulty indices...")
        
        available_cols = [col for col in assessment_columns if col in self.df.columns]
        
        if not available_cols:
            logger.warning("No assessment columns found")
            return pd.DataFrame()
        
        difficulty_data = []
        
        for assessment in available_cols:
            max_score = self.df[assessment].max()
            mean_score = self.df[assessment].mean()
            
            # Difficulty index (higher = easier)
            difficulty = (mean_score / max_score * 100) if max_score > 0 else 0
            
            # Classify difficulty
            if difficulty >= 70:
                category = 'Easy'
            elif difficulty >= 40:
                category = 'Medium'
            else:
                category = 'Hard'
            
            difficulty_data.append({
                'assessment': assessment,
                'mean_score': mean_score,
                'max_score': max_score,
                'difficulty_index': difficulty,
                'difficulty_category': category,
                'std_dev': self.df[assessment].std()
            })
        
        difficulty_df = pd.DataFrame(difficulty_data)
        difficulty_df = difficulty_df.sort_values('difficulty_index')
        
        logger.info(f"Calculated difficulty for {len(difficulty_df)} assessments")
        
        return difficulty_df
    
    def identify_problematic_assessments(self, difficulty_df: pd.DataFrame) -> Dict:
        """
        Identify assessments that are too easy or too hard
        
        Args:
            difficulty_df: DataFrame with difficulty indices
            
        Returns:
            Dictionary with problematic assessments
        """
        logger.info("Identifying problematic assessments...")
        
        too_hard = difficulty_df[difficulty_df['difficulty_index'] < 40]
        too_easy = difficulty_df[difficulty_df['difficulty_index'] > 85]
        
        problematic = {
            'too_hard': too_hard['assessment'].tolist(),
            'too_easy': too_easy['assessment'].tolist(),
            'count_hard': len(too_hard),
            'count_easy': len(too_easy)
        }
        
        logger.info(f"Found {problematic['count_hard']} hard assessments")
        logger.info(f"Found {problematic['count_easy']} easy assessments")
        
        return problematic
    
    def plot_difficulty_distribution(self, difficulty_df: pd.DataFrame,
                                    save_path: Path = None) -> None:
        """Plot assessment difficulty distribution"""
        if difficulty_df.empty:
            logger.warning("No difficulty data to plot")
            return
        
        logger.info("Plotting difficulty distribution...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart of difficulty indices
        colors = difficulty_df['difficulty_category'].map({
            'Easy': 'green',
            'Medium': 'orange',
            'Hard': 'red'
        })
        
        ax1.barh(range(len(difficulty_df)), difficulty_df['difficulty_index'],
                color=colors, alpha=0.7)
        ax1.set_yticks(range(len(difficulty_df)))
        ax1.set_yticklabels(difficulty_df['assessment'])
        ax1.set_xlabel('Difficulty Index (%)', fontweight='bold')
        ax1.set_title('Assessment Difficulty', fontsize=14, fontweight='bold')
        ax1.axvline(x=40, color='red', linestyle='--', alpha=0.5, label='Hard threshold')
        ax1.axvline(x=70, color='green', linestyle='--', alpha=0.5, label='Easy threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Pie chart of difficulty categories
        category_counts = difficulty_df['difficulty_category'].value_counts()
        ax2.pie(category_counts.values, labels=category_counts.index,
               autopct='%1.1f%%', colors=['red', 'orange', 'green'])
        ax2.set_title('Difficulty Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = FIGURES_DIR / "assessment_difficulty.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Difficulty distribution plot saved to: {save_path}")