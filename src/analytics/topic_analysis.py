### src/analytics/topic_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from pathlib import Path
from src.config import FIGURES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TopicAnalysis:
    """Analyze performance by topic/subject area"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def analyze_topic_performance(self, topic_columns: List[str]) -> pd.DataFrame:
        """
        Analyze average performance by topic
        
        Args:
            topic_columns: List of columns representing different topics
            
        Returns:
            DataFrame with topic performance statistics
        """
        logger.info("Analyzing topic-wise performance...")
        
        available_cols = [col for col in topic_columns if col in self.df.columns]
        
        if not available_cols:
            logger.warning("No topic columns found in dataset")
            return pd.DataFrame()
        
        topic_stats = []
        
        for topic in available_cols:
            stats = {
                'topic': topic,
                'mean_score': self.df[topic].mean(),
                'median_score': self.df[topic].median(),
                'std_score': self.df[topic].std(),
                'min_score': self.df[topic].min(),
                'max_score': self.df[topic].max(),
                'pass_rate': (self.df[topic] >= 50).mean() * 100
            }
            topic_stats.append(stats)
        
        stats_df = pd.DataFrame(topic_stats)
        stats_df = stats_df.sort_values('mean_score', ascending=False)
        
        logger.info(f"Analyzed {len(stats_df)} topics")
        
        return stats_df
    
    def identify_weak_topics(self, topic_columns: List[str], 
                            threshold: float = 50) -> List[str]:
        """
        Identify topics where students perform poorly
        
        Args:
            topic_columns: List of topic columns
            threshold: Score threshold for weak performance
            
        Returns:
            List of weak topics
        """
        logger.info(f"Identifying weak topics (threshold: {threshold})...")
        
        available_cols = [col for col in topic_columns if col in self.df.columns]
        
        weak_topics = []
        for topic in available_cols:
            avg_score = self.df[topic].mean()
            if avg_score < threshold:
                weak_topics.append(topic)
        
        logger.info(f"Found {len(weak_topics)} weak topics")
        
        return weak_topics
    
    def plot_topic_performance(self, topic_stats: pd.DataFrame,
                              save_path: Path = None) -> None:
        """Plot topic performance comparison"""
        if topic_stats.empty:
            logger.warning("No topic statistics to plot")
            return
        
        logger.info("Plotting topic performance...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(topic_stats))
        width = 0.35
        
        ax.bar(x - width/2, topic_stats['mean_score'], width, 
               label='Mean Score', alpha=0.8)
        ax.bar(x + width/2, topic_stats['pass_rate'], width,
               label='Pass Rate (%)', alpha=0.8)
        
        ax.set_xlabel('Topic', fontweight='bold')
        ax.set_ylabel('Score / Rate', fontweight='bold')
        ax.set_title('Topic Performance Analysis', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(topic_stats['topic'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = FIGURES_DIR / "topic_performance.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Topic performance plot saved to: {save_path}")