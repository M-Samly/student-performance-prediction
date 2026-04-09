###src/analytics/attendance_correlation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple
from pathlib import Path
from src.config import FIGURES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AttendanceCorrelation:
    """Analyze relationship between attendance and performance"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def calculate_attendance_performance_correlation(self, 
                                                     attendance_col: str,
                                                     performance_col: str) -> Tuple[float, float]:
        """
        Calculate correlation between attendance and performance
        
        Args:
            attendance_col: Attendance column name
            performance_col: Performance/grade column name
            
        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        logger.info("Calculating attendance-performance correlation...")
        
        if attendance_col not in self.df.columns or performance_col not in self.df.columns:
            logger.warning("Required columns not found")
            return 0.0, 1.0
        
        # Remove NaN values
        valid_data = self.df[[attendance_col, performance_col]].dropna()
        
        correlation, p_value = stats.pearsonr(valid_data[attendance_col], 
                                              valid_data[performance_col])
        
        logger.info(f"Correlation: {correlation:.4f} (p-value: {p_value:.4f})")
        
        return correlation, p_value
    
    def analyze_by_attendance_groups(self, attendance_col: str,
                                    performance_col: str) -> pd.DataFrame:
        """
        Analyze performance by attendance groups
        
        Args:
            attendance_col: Attendance column name
            performance_col: Performance column name
            
        Returns:
            DataFrame with group statistics
        """
        logger.info("Analyzing performance by attendance groups...")
        
        if attendance_col not in self.df.columns or performance_col not in self.df.columns:
            logger.warning("Required columns not found")
            return pd.DataFrame()
        
        # Create attendance groups
        self.df['attendance_group'] = pd.cut(
            self.df[attendance_col],
            bins=[0, 50, 75, 90, 100],
            labels=['Poor (<50%)', 'Average (50-75%)', 'Good (75-90%)', 'Excellent (>90%)']
        )
        
        group_stats = self.df.groupby('attendance_group')[performance_col].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('pass_rate', lambda x: (x >= 50).mean() * 100)
        ]).round(2)
        
        logger.info(f"Created {len(group_stats)} attendance groups")
        
        return group_stats
    
    def plot_attendance_vs_performance(self, attendance_col: str,
                                      performance_col: str,
                                      save_path: Path = None) -> None:
        """Plot attendance vs performance relationship"""
        logger.info("Plotting attendance vs performance...")
        
        if attendance_col not in self.df.columns or performance_col not in self.df.columns:
            logger.warning("Required columns not found. Cannot create plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        ax1.scatter(self.df[attendance_col], self.df[performance_col], alpha=0.5)
        
        # Add regression line
        valid_data = self.df[[attendance_col, performance_col]].dropna()
        z = np.polyfit(valid_data[attendance_col], valid_data[performance_col], 1)
        p = np.poly1d(z)
        ax1.plot(valid_data[attendance_col].sort_values(), 
                p(valid_data[attendance_col].sort_values()),
                "r--", alpha=0.8, linewidth=2, label='Trend line')
        
        correlation, _ = self.calculate_attendance_performance_correlation(
            attendance_col, performance_col
        )
        
        ax1.set_xlabel('Attendance (%)', fontweight='bold')
        ax1.set_ylabel('Performance Score', fontweight='bold')
        ax1.set_title(f'Attendance vs Performance\n(Correlation: {correlation:.3f})',
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by attendance groups
        if 'attendance_group' in self.df.columns:
            self.df.boxplot(column=performance_col, by='attendance_group', ax=ax2)
            ax2.set_xlabel('Attendance Group', fontweight='bold')
            ax2.set_ylabel('Performance Score', fontweight='bold')
            ax2.set_title('Performance by Attendance Group', fontsize=14, fontweight='bold')
            plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = FIGURES_DIR / "attendance_vs_performance.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attendance vs performance plot saved to: {save_path}")