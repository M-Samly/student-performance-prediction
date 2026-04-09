### scripts/07_generate_analytics.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.analytics.topic_analysis import TopicAnalysis
from src.analytics.assessment_difficulty import AssessmentDifficulty
from src.analytics.attendance_correlation import AttendanceCorrelation
from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR
from src.utils.file_handler import load_csv, save_csv, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Generate analytics and insights"""
    
    logger.info("="*50)
    logger.info("STEP 7: GENERATING ANALYTICS AND INSIGHTS")
    logger.info("="*50)
    
    # Load featured data
    featured_data_path = PROCESSED_DATA_DIR / "featured_data.csv"
    df = load_csv(featured_data_path)
    
    logger.info(f"Loaded data: {df.shape}")
    
    analytics_results = {}
    
    # ========================================
    # Topic-wise Performance Analysis
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("TOPIC-WISE PERFORMANCE ANALYSIS")
    logger.info("="*50)
    
    # Identify topic/assessment columns
    # ADJUST THESE based on your actual column names
    topic_columns = [col for col in df.columns if 'assignment' in col.lower() or 'quiz' in col.lower()]
    
    if topic_columns:
        topic_analyzer = TopicAnalysis(df)
        
        topic_stats = topic_analyzer.analyze_topic_performance(topic_columns)
        
        if not topic_stats.empty:
            logger.info("\nTopic Performance Statistics:")
            logger.info(topic_stats.to_string())
            
            # Save topic stats
            topic_stats_path = REPORTS_DIR / "topic_performance_stats.csv"
            save_csv(topic_stats, topic_stats_path)
            logger.info(f"Topic stats saved to: {topic_stats_path}")
            
            # Identify weak topics
            weak_topics = topic_analyzer.identify_weak_topics(topic_columns, threshold=50)
            logger.info(f"\nWeak Topics (avg < 50): {weak_topics}")
            
            # Plot topic performance
            topic_analyzer.plot_topic_performance(topic_stats)
            
            analytics_results['topic_analysis'] = {
                'total_topics': len(topic_stats),
                'weak_topics': weak_topics,
                'avg_performance': float(topic_stats['mean_score'].mean())
            }
    else:
        logger.warning("No topic columns found for analysis")
    
    # ========================================
    # Assessment Difficulty Analysis
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("ASSESSMENT DIFFICULTY ANALYSIS")
    logger.info("="*50)
    
    # Assessment columns (can be same as topic columns or different)
    assessment_columns = [col for col in df.columns if 'assignment' in col.lower() 
                         or 'quiz' in col.lower() or 'midterm' in col.lower()]
    
    if assessment_columns:
        difficulty_analyzer = AssessmentDifficulty(df)
        
        difficulty_stats = difficulty_analyzer.calculate_difficulty_index(assessment_columns)
        
        if not difficulty_stats.empty:
            logger.info("\nAssessment Difficulty Statistics:")
            logger.info(difficulty_stats.to_string())
            
            # Save difficulty stats
            difficulty_path = REPORTS_DIR / "assessment_difficulty_stats.csv"
            save_csv(difficulty_stats, difficulty_path)
            logger.info(f"Difficulty stats saved to: {difficulty_path}")
            
            # Identify problematic assessments
            problematic = difficulty_analyzer.identify_problematic_assessments(difficulty_stats)
            logger.info(f"\nProblematic Assessments:")
            logger.info(f"  Too Hard: {problematic['too_hard']}")
            logger.info(f"  Too Easy: {problematic['too_easy']}")
            
            # Plot difficulty distribution
            difficulty_analyzer.plot_difficulty_distribution(difficulty_stats)
            
            analytics_results['difficulty_analysis'] = {
                'total_assessments': len(difficulty_stats),
                'too_hard_count': problematic['count_hard'],
                'too_easy_count': problematic['count_easy'],
                'avg_difficulty': float(difficulty_stats['difficulty_index'].mean())
            }
    else:
        logger.warning("No assessment columns found for difficulty analysis")
    
    # ========================================
    # Attendance vs Performance Analysis
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("ATTENDANCE VS PERFORMANCE ANALYSIS")
    logger.info("="*50)
    
    # ADJUST these column names based on your data
    attendance_col = 'attendance_percentage'
    performance_col = 'final_grade'
    
    # Try to find attendance column
    if attendance_col not in df.columns:
        possible_attendance_cols = [col for col in df.columns if 'attendance' in col.lower()]
        if possible_attendance_cols:
            attendance_col = possible_attendance_cols[0]
            logger.info(f"Using attendance column: {attendance_col}")
    
    # Try to find performance column
    if performance_col not in df.columns:
        possible_performance_cols = [col for col in df.columns if 'grade' in col.lower() or 'score' in col.lower()]
        if possible_performance_cols:
            performance_col = possible_performance_cols[0]
            logger.info(f"Using performance column: {performance_col}")
    
    if attendance_col in df.columns and performance_col in df.columns:
        attendance_analyzer = AttendanceCorrelation(df)
        
        # Calculate correlation
        correlation, p_value = attendance_analyzer.calculate_attendance_performance_correlation(
            attendance_col, performance_col
        )
        
        logger.info(f"\nAttendance-Performance Correlation: {correlation:.4f}")
        logger.info(f"P-value: {p_value:.4f}")
        
        # Determine significance
        if p_value < 0.05:
            significance = "Statistically significant"
        else:
            significance = "Not statistically significant"
        
        logger.info(f"Significance: {significance}")
        
        # Analyze by attendance groups
        group_stats = attendance_analyzer.analyze_by_attendance_groups(
            attendance_col, performance_col
        )
        
        if not group_stats.empty:
            logger.info("\nPerformance by Attendance Group:")
            logger.info(group_stats.to_string())
            
            # Save group stats
            group_stats_path = REPORTS_DIR / "attendance_group_stats.csv"
            group_stats.to_csv(group_stats_path)
            logger.info(f"Group stats saved to: {group_stats_path}")
        
        # Plot attendance vs performance
        attendance_analyzer.plot_attendance_vs_performance(
            attendance_col, performance_col
        )
        
        analytics_results['attendance_analysis'] = {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'significance': significance,
            'relationship': 'positive' if correlation > 0 else 'negative'
        }
    else:
        logger.warning("Required columns for attendance analysis not found")
        logger.warning(f"Looking for: {attendance_col} and {performance_col}")
    
    # ========================================
    # Additional Insights
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("ADDITIONAL INSIGHTS")
    logger.info("="*50)
    
    # Overall statistics
    if 'pass_fail' in df.columns:
        overall_pass_rate = (df['pass_fail'] == 1).mean() * 100
        logger.info(f"Overall Pass Rate: {overall_pass_rate:.2f}%")
        analytics_results['overall_pass_rate'] = float(overall_pass_rate)
    
    # Average scores
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    score_cols = [col for col in numeric_cols if 'score' in col.lower() or 'grade' in col.lower()]
    
    if score_cols:
        avg_scores = df[score_cols].mean()
        logger.info("\nAverage Scores:")
        for col, score in avg_scores.items():
            logger.info(f"  {col}: {score:.2f}")
    
    # Engagement metrics
    if 'engagement_score' in df.columns:
        avg_engagement = df['engagement_score'].mean()
        logger.info(f"\nAverage Engagement Score: {avg_engagement:.2f}")
        analytics_results['avg_engagement'] = float(avg_engagement)
    
    # Risk distribution
    if 'combined_risk_score' in df.columns:
        risk_dist = df['combined_risk_score'].value_counts().sort_index()
        logger.info("\nRisk Score Distribution:")
        logger.info(risk_dist.to_string())
    
    # ========================================
    # Save Analytics Summary
    # ========================================
    analytics_summary_path = REPORTS_DIR / "analytics_summary.json"
    save_json(analytics_results, analytics_summary_path)
    
    logger.info(f"\nAnalytics summary saved to: {analytics_summary_path}")
    logger.info(f"All figures saved to: {FIGURES_DIR}")
    
    logger.info("\n" + "="*50)
    logger.info("ANALYTICS GENERATION COMPLETE")
    logger.info("="*50)

if __name__ == "__main__":
    main()