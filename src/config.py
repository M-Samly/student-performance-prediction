###src/config.py
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODELS_DIR, OUTPUTS_DIR, LOGS_DIR, FIGURES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Data split ratios
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Risk thresholds (based on predicted pass probability)
RISK_THRESHOLDS = {
    'high': 0.5,      # < 50% probability of passing
    'medium': 0.7,    # 50-70% probability
    'low': 0.7        # >= 70% probability
}

# Model parameters
MODEL_PARAMS = {
    'logistic_regression': {
        'random_state': RANDOM_SEED,
        'max_iter': 1000,
        'class_weight': 'balanced'
    },
    'random_forest': {
        'n_estimators': 100,
        'random_state': RANDOM_SEED,
        'class_weight': 'balanced',
        'max_depth': 10,
        'min_samples_split': 5
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'random_state': RANDOM_SEED,
        'learning_rate': 0.1,
        'max_depth': 5
    }
}

# OULAD Dataset specific configurations
STUDENT_ID_COL = 'id_student'
MODULE_COL = 'code_module'
PRESENTATION_COL = 'code_presentation'

# Target variable configuration
TARGET_COL = 'final_result'
PASS_CATEGORIES = ['Pass', 'Distinction']  # These are considered as "Pass"
FAIL_CATEGORIES = ['Fail', 'Withdrawn']    # These are considered as "Fail"

# CSV file names
CSV_FILES = {
    'student_info': 'studentInfo.csv',
    'student_assessment': 'studentAssessment.csv',
    'student_registration': 'studentRegistration.csv',
    'student_vle': 'studentVle.csv',
    'assessments': 'assessments.csv',
    'courses': 'courses.csv',
    'vle': 'vle.csv'
}

# Evaluation metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']