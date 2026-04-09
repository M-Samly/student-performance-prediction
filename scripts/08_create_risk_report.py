### scripts/08_create_risk_report.py
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import REPORTS_DIR, FIGURES_DIR
from src.utils.file_handler import load_csv, load_json, save_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_executive_summary(analytics_results: dict, 
                            prediction_summary: dict) -> str:
    """Create executive summary text"""
    
    summary = f"""
STUDENT PERFORMANCE PREDICTION SYSTEM
EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{"="*70}
OVERVIEW
{"="*70}

This report summarizes the findings from the Student Performance Prediction
System, which analyzes student data to predict outcomes and identify at-risk
students for early intervention.

Total Students Analyzed: {prediction_summary.get('total_students', 'N/A')}
Overall Pass Rate: {analytics_results.get('overall_pass_rate', 'N/A'):.2f}%

{"="*70}
PREDICTION RESULTS
{"="*70}

Predicted to Pass: {prediction_summary.get('predicted_pass', 'N/A')}
Predicted to Fail: {prediction_summary.get('predicted_fail', 'N/A')}

Model Accuracy: {prediction_summary.get('accuracy', 0)*100:.2f}%

Risk Distribution:
  - High Risk: {prediction_summary.get('risk_distribution', {}).get('High', 0)} students
  - Medium Risk: {prediction_summary.get('risk_distribution', {}).get('Medium', 0)} students
  - Low Risk: {prediction_summary.get('risk_distribution', {}).get('Low', 0)} students

At-Risk Students: {prediction_summary.get('at_risk_count', 0)} 
({prediction_summary.get('at_risk_percentage', 0):.1f}%)

{"="*70}
KEY INSIGHTS
{"="*70}
"""
    
    # Add attendance insights if available
    if 'attendance_analysis' in analytics_results:
        att_analysis = analytics_results['attendance_analysis']
        summary += f"""
Attendance-Performance Correlation: {att_analysis.get('correlation', 0):.3f}
Relationship: {att_analysis.get('relationship', 'N/A').capitalize()}
Statistical Significance: {att_analysis.get('significance', 'N/A')}
"""
    
    # Add topic insights if available
    if 'topic_analysis' in analytics_results:
        topic_analysis = analytics_results['topic_analysis']
        summary += f"""
Total Topics Analyzed: {topic_analysis.get('total_topics', 0)}
Average Performance: {topic_analysis.get('avg_performance', 0):.2f}%
Weak Topics Identified: {len(topic_analysis.get('weak_topics', []))}
"""
    
    # Add difficulty insights if available
    if 'difficulty_analysis' in analytics_results:
        diff_analysis = analytics_results['difficulty_analysis']
        summary += f"""
Total Assessments: {diff_analysis.get('total_assessments', 0)}
Average Difficulty: {diff_analysis.get('avg_difficulty', 0):.2f}%
Assessments Too Hard: {diff_analysis.get('too_hard_count', 0)}
Assessments Too Easy: {diff_analysis.get('too_easy_count', 0)}
"""
    
    summary += f"""
{"="*70}
RECOMMENDATIONS
{"="*70}

1. IMMEDIATE ACTIONS:
   - Provide targeted support to {prediction_summary.get('at_risk_count', 0)} at-risk students
   - Focus on high-risk students for intensive intervention
   - Monitor medium-risk students closely

2. ATTENDANCE IMPROVEMENT:
"""
    
    if 'attendance_analysis' in analytics_results:
        if analytics_results['attendance_analysis'].get('correlation', 0) > 0.5:
            summary += """   - Strong correlation found between attendance and performance
   - Implement attendance monitoring and incentive programs
   - Contact students with low attendance rates
"""
        else:
            summary += """   - Moderate correlation with performance
   - Continue monitoring attendance patterns
"""
    
    summary += """
3. CURRICULUM ADJUSTMENTS:
"""
    
    if 'topic_analysis' in analytics_results:
        weak_topics = analytics_results['topic_analysis'].get('weak_topics', [])
        if weak_topics:
            summary += f"""   - Focus on improving instruction in weak topics
   - Provide additional resources for: {', '.join(weak_topics[:3])}
"""
    
    if 'difficulty_analysis' in analytics_results:
        summary += """   - Review and adjust difficulty of assessments
   - Balance assessment difficulty for better learning outcomes
"""
    
    summary += """
4. ONGOING MONITORING:
   - Continue tracking student performance metrics
   - Refine prediction models with new data
   - Implement feedback mechanisms for interventions

{"="*70}
"""
    
    return summary

def create_detailed_risk_report(at_risk_students: pd.DataFrame,
                               prediction_report: pd.DataFrame) -> pd.DataFrame:
    """Create detailed report for at-risk students"""
    
    logger.info("Creating detailed risk report...")
    
    # Merge additional information if available
    detailed_report = at_risk_students.copy()
    
    # Add recommendations based on risk level
    def get_recommendation(row):
        if row['risk_level'] == 'High':
            return "URGENT: Schedule immediate meeting, provide tutoring, monitor daily"
        elif row['risk_level'] == 'Medium':
            return "PRIORITY: Offer additional support, weekly check-ins"
        else:
            return "WATCH: Monitor progress, provide resources as needed"
    
    detailed_report['recommendation'] = detailed_report.apply(get_recommendation, axis=1)
    
    # Sort by risk and probability
    detailed_report = detailed_report.sort_values(
        ['risk_level', 'pass_probability'],
        ascending=[False, True]
    )
    
    logger.info(f"Detailed report created for {len(detailed_report)} students")
    
    return detailed_report

def main():
    """Create comprehensive risk and intervention report"""
    
    logger.info("="*50)
    logger.info("STEP 8: CREATING RISK REPORT")
    logger.info("="*50)
    
    # Load prediction results
    prediction_report_path = REPORTS_DIR / "prediction_report_full.csv"
    at_risk_path = REPORTS_DIR / "at_risk_students.csv"
    prediction_summary_path = REPORTS_DIR / "prediction_summary.json"
    analytics_summary_path = REPORTS_DIR / "analytics_summary.json"
    
    # Check if required files exist
    if not prediction_report_path.exists():
        logger.error("Prediction report not found. Run script 06 first.")
        return
    
    prediction_report = load_csv(prediction_report_path)
    at_risk_students = load_csv(at_risk_path) if at_risk_path.exists() else pd.DataFrame()
    prediction_summary = load_json(prediction_summary_path) if prediction_summary_path.exists() else {}
    analytics_results = load_json(analytics_summary_path) if analytics_summary_path.exists() else {}
    
    # ========================================
    # Create Executive Summary
    # ========================================
    logger.info("\nCreating executive summary...")
    
    executive_summary = create_executive_summary(analytics_results, prediction_summary)
    
    # Save executive summary
    exec_summary_path = REPORTS_DIR / "executive_summary.txt"
    with open(exec_summary_path, 'w') as f:
        f.write(executive_summary)
    
    logger.info(f"Executive summary saved to: {exec_summary_path}")
    
    # Print to console
    print("\n" + executive_summary)
    
    # ========================================
    # Create Detailed Risk Report
    # ========================================
    if not at_risk_students.empty:
        logger.info("\nCreating detailed risk report...")
        
        detailed_risk_report = create_detailed_risk_report(
            at_risk_students, prediction_report
        )
        
        # Save detailed report
        detailed_report_path = REPORTS_DIR / "detailed_risk_report.csv"
        save_csv(detailed_risk_report, detailed_report_path)
        
        logger.info(f"Detailed risk report saved to: {detailed_report_path}")
        
        # Create summary by risk level
        # Create summary by risk level
        risk_summary = detailed_risk_report.groupby('risk_level').agg({
            'prediction': 'count',  # Use 'prediction' instead
            'pass_probability': ['mean', 'min', 'max']
        }).round(3)

        # Rename the count column for clarity
        risk_summary.columns = ['count', 'avg_prob', 'min_prob', 'max_prob']
        
        logger.info("\nRisk Level Summary:")
        logger.info(risk_summary.to_string())
        
    # ========================================
    # Create Action Items List
    # ========================================
    logger.info("\nCreating action items...")
    
    action_items = []
    
    # High-risk students
    if not at_risk_students.empty:
        high_risk = at_risk_students[at_risk_students['risk_level'] == 'High']
        if len(high_risk) > 0:
            action_items.append({
                'priority': 'URGENT',
                'category': 'Student Intervention',
                'action': f'Schedule meetings with {len(high_risk)} high-risk students',
                'responsible': 'Academic Advisors',
                'deadline': '1 week'
            })
        
        medium_risk = at_risk_students[at_risk_students['risk_level'] == 'Medium']
        if len(medium_risk) > 0:
            action_items.append({
                'priority': 'HIGH',
                'category': 'Student Support',
                'action': f'Provide additional resources to {len(medium_risk)} medium-risk students',
                'responsible': 'Course Instructors',
                'deadline': '2 weeks'
            })
    
    # Attendance issues
    if 'attendance_analysis' in analytics_results:
        if analytics_results['attendance_analysis'].get('correlation', 0) > 0.5:
            action_items.append({
                'priority': 'HIGH',
                'category': 'Attendance',
                'action': 'Implement attendance improvement program',
                'responsible': 'Student Affairs',
                'deadline': '1 month'
            })
    
    # Curriculum adjustments
    if 'topic_analysis' in analytics_results:
        weak_topics = analytics_results['topic_analysis'].get('weak_topics', [])
        if weak_topics:
            action_items.append({
                'priority': 'MEDIUM',
                'category': 'Curriculum',
                'action': f'Review and enhance teaching materials for weak topics',
                'responsible': 'Department Head',
                'deadline': '1 month'
            })
    
    # Assessment review
    if 'difficulty_analysis' in analytics_results:
        if analytics_results['difficulty_analysis'].get('too_hard_count', 0) > 0:
            action_items.append({
                'priority': 'MEDIUM',
                'category': 'Assessment',
                'action': 'Review and adjust difficulty of challenging assessments',
                'responsible': 'Course Coordinator',
                'deadline': '3 weeks'
            })
    
    # Save action items
    if action_items:
        action_items_df = pd.DataFrame(action_items)
        action_items_path = REPORTS_DIR / "action_items.csv"
        save_csv(action_items_df, action_items_path)
        
        logger.info(f"\nAction items saved to: {action_items_path}")
        logger.info(f"Total action items: {len(action_items)}")
    
    # ========================================
    # Create Final Report Package Summary
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("REPORT PACKAGE SUMMARY")
    logger.info("="*50)
    
    report_files = {
        'Executive Summary': exec_summary_path,
        'Full Prediction Report': prediction_report_path,
        'At-Risk Students': at_risk_path,
        'Detailed Risk Report': REPORTS_DIR / "detailed_risk_report.csv" if not at_risk_students.empty else None,
        'Action Items': REPORTS_DIR / "action_items.csv" if action_items else None,
        'Analytics Summary': analytics_summary_path,
        'Model Comparison': REPORTS_DIR / "final_model_comparison.csv"
    }
    
    logger.info("\nGenerated Reports:")
    for report_name, report_path in report_files.items():
        if report_path and report_path.exists():
            logger.info(f"  [OK] {report_name}: {report_path}")
        else:
            logger.info(f"  [--] {report_name}: Not available")
    
    logger.info(f"\nAll reports saved to: {REPORTS_DIR}")
    logger.info(f"All figures saved to: {FIGURES_DIR}")
    
    logger.info("\n" + "="*50)
    logger.info("RISK REPORT CREATION COMPLETE")
    logger.info("="*50)

if __name__ == "__main__":
    main()