# Student Performance Prediction System (OULAD)

An end-to-end **Data Science / Machine Learning pipeline** that predicts student outcomes (**Pass vs Fail/Withdrawn**) using the **Open University Learning Analytics Dataset (OULAD)**.  
The project includes: data integration, preprocessing, feature engineering, baseline + improved models, evaluation (ROC/PR/confusion matrix), explainability (SHAP), and risk-level reporting.

> This repository is a **student capstone project** (CS5998). It is intended for learning and evaluation purposes, not production deployment.

---

## 1) Project Overview

### Goal
Build a system that:
- Predicts whether a student will **Pass** (Pass/Distinction) or **Fail** (Fail/Withdrawn)
- Produces **risk levels** (High/Medium/Low) based on predicted pass probability
- Generates evaluation plots and reports for reproducibility

### Dataset
- **OULAD (Open University Learning Analytics Dataset)**
- Citation: Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset*. Scientific Data, 4, 170171.

Dataset files used:
- `studentInfo.csv`
- `studentVle.csv`
- `studentAssessment.csv`
- `studentRegistration.csv`
- `assessments.csv`
- `courses.csv`
- `vle.csv`

---

## 2) Repository Structure (Main Folders)

```
student-performance-prediction/
├── data/
│   ├── raw/                # Put OULAD CSV files here (NOT committed to GitHub recommended)
│   └── processed/          # Auto-generated intermediate datasets + splits
├── scripts/                # Pipeline scripts (01 to 08)
├── src/                    # Core code (data, models, evaluation, analytics, utils)
├── models/                 # Auto-saved trained models (.pkl) + metadata
├── outputs/
│   ├── figures/            # Plots: ROC, PR, confusion matrices, SHAP, EDA
│   ├── reports/            # CSV/JSON/TXT reports (predictions, risk lists, comparisons)
│   └── logs/               # Execution logs
├── notebooks/              # Optional exploration notebooks
├── docs/                   # milestone docs, architecture diagram, final report
├── run_all.py              # Runs full pipeline end-to-end
└── requirements.txt
```

---

## 3) Setup Instructions

### 3.1 Requirements
- Python **3.10+** (tested with Python **3.11**)
- Recommended RAM: **8GB+** (OULAD has large interaction logs)
- OS: Windows / Linux / macOS

### 3.2 Create a Virtual Environment

#### Windows (PowerShell)
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3.3 Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4) Download and Place the Dataset (OULAD)

### IMPORTANT (GitHub note)
OULAD files can be large. It's recommended **NOT** to push `data/raw/` to GitHub.  
Add to `.gitignore`:
```
data/raw/
data/processed/
models/
outputs/
venv/
```
(You can still keep small sample outputs if needed.)

### Steps
1. Download OULAD from Kaggle or official source  - https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad
2. Copy the CSV files into:

```
data/raw/
```

### Expected filenames
Make sure these exact names exist in `data/raw/`:
- `studentInfo.csv`
- `studentVle.csv`
- `studentAssessment.csv`
- `studentRegistration.csv`
- `assessments.csv`
- `courses.csv`
- `vle.csv`

---

## 5) How to Run the Project (Step-by-Step)

You can run the full pipeline or each step individually.

## Option A — Run Everything (Recommended)
From the project root:

```bash
python run_all.py
```

This runs scripts 01 → 08 in sequence and generates:
- processed datasets
- trained models
- evaluation figures
- prediction + risk reports
- logs

---

## Option B — Run Each Script Manually

### Step 1: Load + Validate + Merge Data
```bash
python scripts/01_load_and_validate_data.py
```
Outputs:
- `data/processed/merged_raw_data.csv`
- `data/processed/validation_report.json`

### Step 2: Preprocess + Feature Engineering + Split Data
```bash
python scripts/02_preprocess_data.py
```
Outputs:
- `data/processed/cleaned_data.csv`
- `data/processed/featured_data.csv`
- `data/processed/train_test_split/train.csv`
- `data/processed/train_test_split/validation.csv`
- `data/processed/train_test_split/test.csv`
- `data/processed/preprocessing_info.json`

### Step 3: Train Baseline Model (Logistic Regression)
```bash
python scripts/03_train_baseline.py
```
Outputs:
- `models/baseline/logistic_regression.pkl`
- `models/baseline/model_metadata.json`
- `models/baseline/feature_importance.csv`

### Step 4: Train Improved Models (Random Forest + Gradient Boosting)
```bash
python scripts/04_train_improved_models.py
```
Outputs:
- `models/improved/random_forest.pkl`
- `models/improved/gradient_boosting.pkl`
- `models/improved/model_comparison.csv`
- `models/improved/best_model_info.json`

### Step 5: Evaluate Models + Generate Plots (ROC/PR/Confusion + SHAP)
```bash
python scripts/05_evaluate_models.py
```
Outputs (examples):
- `outputs/figures/*_confusion_matrix.png`
- `outputs/figures/*_roc_curve.png`
- `outputs/figures/*_pr_curve.png`
- `outputs/figures/*_shap_summary.png`
- `outputs/reports/final_model_comparison.csv`
- `outputs/reports/best_model_evaluation.json`

### Step 6: Generate Predictions + Risk Levels
```bash
python scripts/06_generate_predictions.py
```
Outputs:
- `outputs/reports/prediction_report_full.csv`
- `outputs/reports/at_risk_students.csv`
- `outputs/reports/prediction_summary.json`

### Step 7: Generate Analytics (Engagement / Assessment / Module breakdown)
```bash
python scripts/07_generate_analytics.py
```
Outputs depend on the analytics implementation, but typically include:
- analytics summary JSON and supporting CSV/PNG outputs in `outputs/reports/` and `outputs/figures/`

### Step 8: Create Risk Report Package (Executive Summary + Action Items)
```bash
python scripts/08_create_risk_report.py
```
Outputs:
- `outputs/reports/executive_summary.txt`
- `outputs/reports/detailed_risk_report.csv`
- `outputs/reports/action_items.csv` (if generated)

---

## 6) Where to Find Results

### Main folders
- Figures: `outputs/figures/`
- Reports: `outputs/reports/`
- Logs: `outputs/logs/`
- Trained models: `models/`

### Common plots you should see
- `target_distribution.png`
- `correlation_matrix.png`
- `feature_distributions.png`
- `boxplots_by_target.png`
- `*_confusion_matrix.png`
- `*_roc_curve.png`
- `*_pr_curve.png`
- `*_shap_summary.png` and `*_shap_importance.png`

---

## 7) Running Notebooks (Optional)
If you want interactive exploration:

```bash
jupyter lab
```

Notebooks are under:
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_data_preprocessing.ipynb`
- `notebooks/03_baseline_model.ipynb`
- `notebooks/04_improved_models.ipynb`
- `notebooks/05_model_evaluation.ipynb`
- `notebooks/06_explainability.ipynb`

---

## 8) Configuration
Main configuration is in:
- `src/config.py`

Includes:
- directory paths
- random seed
- model hyperparameters
- OULAD column names
- risk thresholds

---

## 9) Reproducibility
- Fixed random seed: `RANDOM_SEED = 42`
- Train/validation/test splits saved as CSV
- Model artifacts saved as `.pkl`
- Evaluation reports stored as JSON/CSV

To reproduce all outputs, run:
```bash
python run_all.py
```

---

## 10) Ethical / Responsible Use (Student Note)
This system is intended as **decision support**. In real deployments, educational data requires:
- formal approval processes
- strict privacy controls
- fairness audits (error rates across gender/disability/IMD band, etc.)

---

## 11) Troubleshooting

### Problem: "File not found" for CSV files
- Ensure OULAD CSVs exist in `data/raw/` with the **exact filenames** listed above.

### Problem: SHAP errors / slow runtime
- SHAP can be slower on large datasets. The evaluation script uses sampling for SHAP.
- Ensure `shap` is installed:
```bash
pip install shap
```

### Problem: Memory issues on `studentVle.csv`
- Close other applications
- Use a machine with more RAM
- (Optional future improvement) implement chunked processing

---
## Author
**Samly M.F.M. (258738G)**  
Capstone project repository for CS5998