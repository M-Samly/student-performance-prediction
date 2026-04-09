import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Create figure
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
color_data = '#E3F2FD'
color_process = '#FFF9C4'
color_model = '#C8E6C9'
color_output = '#FFCCBC'
color_analytics = '#E1BEE7'

# Title
ax.text(5, 13.5, 'Student Performance Prediction System Architecture', 
        fontsize=20, fontweight='bold', ha='center')
ax.text(5, 13, 'Open University Learning Analytics Dataset (OULAD)', 
        fontsize=12, ha='center', style='italic')

# ============================================================================
# LAYER 1: DATA SOURCES
# ============================================================================
y_start = 11.5

ax.text(5, y_start + 0.5, 'LAYER 1: DATA SOURCES', 
        fontsize=14, fontweight='bold', ha='center')

data_sources = [
    ('studentInfo.csv\n(Demographics)', 0.5),
    ('studentAssessment.csv\n(Scores)', 2),
    ('studentVle.csv\n(VLE Activity)', 3.5),
    ('studentRegistration.csv\n(Enrollment)', 5),
    ('assessments.csv\n(Assessment Info)', 6.5),
    ('courses.csv\n(Course Info)', 8),
    ('vle.csv\n(VLE Resources)', 9.5)
]

for name, x in data_sources:
    box = FancyBboxPatch((x-0.4, y_start-0.4), 0.8, 0.6,
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color_data, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y_start-0.1, name, fontsize=7, ha='center', va='center')

# ============================================================================
# LAYER 2: DATA INTEGRATION
# ============================================================================
y_layer2 = 9.5

# Arrow from Layer 1 to Layer 2
for x in [1, 2.75, 4.25, 5.75, 7.25, 8.75]:
    arrow = FancyArrowPatch((x, y_start-0.5), (5, y_layer2+0.6),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=1.5, color='gray', alpha=0.6)
    ax.add_patch(arrow)

ax.text(5, y_layer2 + 1, 'LAYER 2: DATA INTEGRATION & VALIDATION', 
        fontsize=14, fontweight='bold', ha='center')

# Data Loader
box = FancyBboxPatch((1.5, y_layer2-0.4), 3, 0.8,
                     boxstyle="round,pad=0.1", 
                     edgecolor='black', facecolor=color_process, linewidth=2)
ax.add_patch(box)
ax.text(3, y_layer2+0.2, 'Data Loader', fontsize=10, fontweight='bold', ha='center')
ax.text(3, y_layer2-0.1, 'Merge on (student, module, presentation)\nAggregate VLE & Assessment data', 
        fontsize=7, ha='center', va='center')

# Data Validator
box = FancyBboxPatch((5.5, y_layer2-0.4), 3, 0.8,
                     boxstyle="round,pad=0.1", 
                     edgecolor='black', facecolor=color_process, linewidth=2)
ax.add_patch(box)
ax.text(7, y_layer2+0.2, 'Data Validator', fontsize=10, fontweight='bold', ha='center')
ax.text(7, y_layer2-0.1, 'Check missing values\nValidate data quality\nVerify distributions', 
        fontsize=7, ha='center', va='center')

# ============================================================================
# LAYER 3: DATA PREPROCESSING
# ============================================================================
y_layer3 = 7.5

# Arrow from Layer 2 to Layer 3
arrow = FancyArrowPatch((5, y_layer2-0.5), (5, y_layer3+0.6),
                       arrowstyle='->', mutation_scale=20, 
                       linewidth=2, color='black')
ax.add_patch(arrow)

ax.text(5, y_layer3 + 1, 'LAYER 3: PREPROCESSING & FEATURE ENGINEERING', 
        fontsize=14, fontweight='bold', ha='center')

preprocessing_steps = [
    ('Data Cleaning\n• Handle missing\n• Remove duplicates\n• Create target', 1.5),
    ('Feature Engineering\n• Engagement (VLE)\n• Assessment\n• Registration\n• Risk indicators', 3.5),
    ('Encoding\n• Categorical vars\n• Label encoding\n• One-hot encoding', 5.5),
    ('Normalization\n• StandardScaler\n• Numeric features\n• Save scaler', 7.5)
]

for name, x in preprocessing_steps:
    box = FancyBboxPatch((x-0.8, y_layer3-0.4), 1.6, 0.8,
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color_process, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y_layer3, name, fontsize=7, ha='center', va='center')

# Data Split
box = FancyBboxPatch((9-0.5, y_layer3-0.4), 1, 0.8,
                     boxstyle="round,pad=0.05", 
                     edgecolor='black', facecolor=color_process, linewidth=1.5)
ax.add_patch(box)
ax.text(9.5, y_layer3, 'Data Split\n• Train 70%\n• Val 10%\n• Test 20%', 
        fontsize=7, ha='center', va='center')

# ============================================================================
# LAYER 4: MODEL TRAINING
# ============================================================================
y_layer4 = 5.5

# Arrow from Layer 3 to Layer 4
arrow = FancyArrowPatch((5, y_layer3-0.5), (5, y_layer4+0.6),
                       arrowstyle='->', mutation_scale=20, 
                       linewidth=2, color='black')
ax.add_patch(arrow)

ax.text(5, y_layer4 + 1, 'LAYER 4: MODEL TRAINING & SELECTION', 
        fontsize=14, fontweight='bold', ha='center')

models = [
    ('Baseline Model\nLogistic\nRegression', 2),
    ('Improved Model 1\nRandom\nForest', 5),
    ('Improved Model 2\nGradient\nBoosting', 8)
]

for name, x in models:
    box = FancyBboxPatch((x-0.9, y_layer4-0.4), 1.8, 0.8,
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color_model, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y_layer4, name, fontsize=9, ha='center', va='center', fontweight='bold')

# ============================================================================
# LAYER 5: EVALUATION
# ============================================================================
y_layer5 = 3.5

# Arrow from Layer 4 to Layer 5
for x in [2, 5, 8]:
    arrow = FancyArrowPatch((x, y_layer4-0.5), (5, y_layer5+0.6),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=1.5, color='gray', alpha=0.6)
    ax.add_patch(arrow)

ax.text(5, y_layer5 + 1, 'LAYER 5: EVALUATION & EXPLAINABILITY', 
        fontsize=14, fontweight='bold', ha='center')

eval_components = [
    ('Metrics\n• Accuracy\n• Precision/Recall\n• F1-Score\n• ROC-AUC', 1.5),
    ('Visualization\n• Confusion Matrix\n• ROC Curves\n• Feature Importance', 4),
    ('Explainability\n• SHAP Analysis\n• Feature Impact\n• Model Insights', 6.5),
    ('Model Selection\n• Compare Models\n• Select Best\n• Save Best Model', 9)
]

for name, x in eval_components:
    box = FancyBboxPatch((x-0.7, y_layer5-0.4), 1.4, 0.8,
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color_output, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y_layer5, name, fontsize=7, ha='center', va='center')

# ============================================================================
# LAYER 6: PREDICTION & ANALYTICS
# ============================================================================
y_layer6 = 1.5

# Arrow from Layer 5 to Layer 6
arrow = FancyArrowPatch((5, y_layer5-0.5), (5, y_layer6+0.6),
                       arrowstyle='->', mutation_scale=20, 
                       linewidth=2, color='black')
ax.add_patch(arrow)

ax.text(5, y_layer6 + 1, 'LAYER 6: PREDICTION & ANALYTICS', 
        fontsize=14, fontweight='bold', ha='center')

prediction_components = [
    ('Risk Prediction\n• Pass/Fail\n• Risk Levels\n(High/Med/Low)', 1.5),
    ('VLE Analytics\n• Engagement\n• Activity Patterns\n• Performance', 3.5),
    ('Assessment\nAnalytics\n• Scores\n• Difficulty\n• Trends', 5.5),
    ('Intervention\nReports\n• At-Risk Students\n• Action Items\n• Recommendations', 7.5)
]

for name, x in prediction_components:
    box = FancyBboxPatch((x-0.8, y_layer6-0.4), 1.6, 0.8,
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color_analytics, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y_layer6, name, fontsize=7, ha='center', va='center')

# ============================================================================
# OUTPUTS
# ============================================================================
y_output = 0.3

# Arrow from Layer 6 to Outputs
arrow = FancyArrowPatch((5, y_layer6-0.5), (5, y_output+0.3),
                       arrowstyle='->', mutation_scale=20, 
                       linewidth=2, color='black')
ax.add_patch(arrow)

output_box = FancyBboxPatch((0.5, y_output-0.2), 9, 0.4,
                           boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor='#FFFDE7', linewidth=2)
ax.add_patch(output_box)

ax.text(5, y_output+0.05, 'OUTPUTS: Reports | Visualizations | Models | Predictions | Executive Summary', 
        fontsize=10, ha='center', va='center', fontweight='bold')

# ============================================================================
# LEGEND
# ============================================================================
legend_y = 12.5
legend_elements = [
    mpatches.Patch(facecolor=color_data, edgecolor='black', label='Data Layer'),
    mpatches.Patch(facecolor=color_process, edgecolor='black', label='Processing Layer'),
    mpatches.Patch(facecolor=color_model, edgecolor='black', label='Model Layer'),
    mpatches.Patch(facecolor=color_output, edgecolor='black', label='Evaluation Layer'),
    mpatches.Patch(facecolor=color_analytics, edgecolor='black', label='Analytics Layer'),
]

ax.legend(handles=legend_elements, loc='upper left', 
         bbox_to_anchor=(0, 0.98), fontsize=8, ncol=5)

# Add metadata
ax.text(0.1, 0.05, 'Technology Stack: Python | scikit-learn | pandas | SHAP | matplotlib', 
        fontsize=7, style='italic', color='gray')
ax.text(9.9, 0.05, 'Student Performance Prediction System v1.0', 
        fontsize=7, ha='right', style='italic', color='gray')

plt.tight_layout()

# Save
output_path = 'docs/milestone_2/architecture_diagram.png'
from pathlib import Path
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Architecture diagram saved to: {output_path}")

# Also save high-res version
plt.savefig('docs/milestone_2/architecture_diagram_highres.png', 
           dpi=600, bbox_inches='tight', facecolor='white')
print(f"High-resolution diagram saved to: docs/milestone_2/architecture_diagram_highres.png")

plt.show()