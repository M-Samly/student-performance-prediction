### scripts/06_generate_predictions.py
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predictor import StudentPredictor
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.utils.file_handler import load_csv, load_model, load_json, save_csv, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Generate predictions and risk classifications"""
    
    logger.info("="*50)
    logger.info("STEP 6: GENERATING PREDICTIONS")
    logger.info("="*50)
    
    # Load best model info
    best_model_info_path = REPORTS_DIR / "best_model_evaluation.json"
    
    if not best_model_info_path.exists():
        logger.warning("Best model info not found. Using Random Forest as default.")
        best_model_name = "Random_Forest"
        model_path = MODELS_DIR / "improved" / "random_forest.pkl"
    else:
        best_model_info = load_json(best_model_info_path)
        best_model_name = best_model_info['best_model']
        
        # Map model name to path
        if 'Logistic' in best_model_name:
            model_path = MODELS_DIR / "baseline" / "logistic_regression.pkl"
        elif 'Random' in best_model_name:
            model_path = MODELS_DIR / "improved" / "random_forest.pkl"
        elif 'Gradient' in best_model_name:
            model_path = MODELS_DIR / "improved" / "gradient_boosting.pkl"
        else:
            model_path = MODELS_DIR / "improved" / "random_forest.pkl"
    
    logger.info(f"Using model: {best_model_name}")
    logger.info(f"Loading from: {model_path}")
    
    # Load model
    model = load_model(model_path)
    
    # Load test data
    splits_dir = PROCESSED_DATA_DIR / "train_test_split"
    test_data = load_csv(splits_dir / "test.csv")
    
    # Check if student_id exists in original data
    featured_data = load_csv(PROCESSED_DATA_DIR / "featured_data.csv")
    
    # Prepare data
    X_test = test_data.drop('pass_fail', axis=1)
    y_test = test_data['pass_fail']
    
    # Try to get student IDs
    student_ids = None
    if 'student_id' in featured_data.columns:
        # Match indices
        student_ids = featured_data.loc[test_data.index, 'student_id'] if 'student_id' in featured_data.columns else None
    
    logger.info(f"Generating predictions for {len(X_test)} students...")
    
    # Initialize predictor
    predictor = StudentPredictor(model)
    
    # Generate predictions
    predictions, probabilities = predictor.predict_outcomes(X_test)
    risk_levels = predictor.classify_risk_level(probabilities)
    
    # Create prediction report
    prediction_report = predictor.create_prediction_report(X_test, student_ids)
    
    # Add actual outcomes for test set
    prediction_report['actual_outcome'] = y_test.values
    prediction_report['actual_outcome_label'] = ['Pass' if y == 1 else 'Fail' for y in y_test]
    prediction_report['correct_prediction'] = (predictions == y_test.values).astype(int)
    
    # Save full prediction report
    full_report_path = REPORTS_DIR / "prediction_report_full.csv"
    save_csv(prediction_report, full_report_path)
    logger.info(f"Full prediction report saved to: {full_report_path}")
    
    # Get at-risk students
    at_risk_students = predictor.get_at_risk_students(prediction_report, ['High', 'Medium'])
    
    # Save at-risk students report
    at_risk_path = REPORTS_DIR / "at_risk_students.csv"
    save_csv(at_risk_students, at_risk_path)
    logger.info(f"At-risk students report saved to: {at_risk_path}")
    
    # Generate summary statistics
    summary_stats = {
        'total_students': len(prediction_report),
        'predicted_pass': int((predictions == 1).sum()),
        'predicted_fail': int((predictions == 0).sum()),
        'risk_distribution': {
            'High': int((risk_levels == 'High').sum()),
            'Medium': int((risk_levels == 'Medium').sum()),
            'Low': int((risk_levels == 'Low').sum())
        },
        'accuracy': float((predictions == y_test.values).mean()),
        'at_risk_count': len(at_risk_students),
        'at_risk_percentage': float(len(at_risk_students) / len(prediction_report) * 100)
    }
    
    # Save summary
    summary_path = REPORTS_DIR / "prediction_summary.json"
    save_json(summary_stats, summary_path)
    
    logger.info("\n" + "="*50)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total Students: {summary_stats['total_students']}")
    logger.info(f"Predicted Pass: {summary_stats['predicted_pass']}")
    logger.info(f"Predicted Fail: {summary_stats['predicted_fail']}")
    logger.info(f"\nRisk Distribution:")
    logger.info(f"  High Risk: {summary_stats['risk_distribution']['High']}")
    logger.info(f"  Medium Risk: {summary_stats['risk_distribution']['Medium']}")
    logger.info(f"  Low Risk: {summary_stats['risk_distribution']['Low']}")
    logger.info(f"\nAt-Risk Students: {summary_stats['at_risk_count']} ({summary_stats['at_risk_percentage']:.1f}%)")
    logger.info(f"Prediction Accuracy: {summary_stats['accuracy']:.4f}")
    
    logger.info(f"\nSummary saved to: {summary_path}")
    
    logger.info("\n" + "="*50)
    logger.info("PREDICTION GENERATION COMPLETE")
    logger.info("="*50)

if __name__ == "__main__":
    main()