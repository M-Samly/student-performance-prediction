### scripts/create_sample_data.py
import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Number of students
n_students = 200

print("Creating sample data...")
print("="*60)

# Create student_records.csv
print("Creating student_records.csv...")
student_records = pd.DataFrame({
    'student_id': [f'S{i:04d}' for i in range(1, n_students + 1)],
    'age': np.random.randint(18, 25, n_students),
    'gender': np.random.choice(['M', 'F'], n_students),
    'program': np.random.choice(['CS', 'IT', 'DS'], n_students)
})

# Create attendance.csv
print("Creating attendance.csv...")
attendance = pd.DataFrame({
    'student_id': [f'S{i:04d}' for i in range(1, n_students + 1)],
    'attendance_percentage': np.random.uniform(40, 100, n_students).round(2)
})

# Create assessments.csv
print("Creating assessments.csv...")
assessments = pd.DataFrame({
    'student_id': [f'S{i:04d}' for i in range(1, n_students + 1)],
    'assignment_1': np.random.uniform(30, 100, n_students).round(2),
    'assignment_2': np.random.uniform(30, 100, n_students).round(2),
    'assignment_3': np.random.uniform(30, 100, n_students).round(2),
    'quiz_1': np.random.uniform(40, 100, n_students).round(2),
    'quiz_2': np.random.uniform(40, 100, n_students).round(2),
    'quiz_3': np.random.uniform(40, 100, n_students).round(2),
    'midterm_score': np.random.uniform(25, 100, n_students).round(2)
})

# Create exam_results.csv
# Make final grade correlated with attendance and assessments
print("Creating exam_results.csv...")

attendance_effect = attendance['attendance_percentage'] * 0.3
assessment_effect = assessments[['assignment_1', 'assignment_2', 'assignment_3']].mean(axis=1) * 0.4
midterm_effect = assessments['midterm_score'] * 0.3
noise = np.random.normal(0, 10, n_students)

final_grade = attendance_effect + assessment_effect + midterm_effect + noise
final_grade = np.clip(final_grade, 0, 100).round(2)  # Ensure grades are between 0-100

exam_results = pd.DataFrame({
    'student_id': [f'S{i:04d}' for i in range(1, n_students + 1)],
    'final_grade': final_grade
})

# Create data/raw directory
print("\nCreating data/raw directory...")
Path('data/raw').mkdir(parents=True, exist_ok=True)

# Save all files
print("\nSaving CSV files...")
student_records.to_csv('data/raw/student_records.csv', index=False)
print("✓ Saved: data/raw/student_records.csv")

attendance.to_csv('data/raw/attendance.csv', index=False)
print("✓ Saved: data/raw/attendance.csv")

assessments.to_csv('data/raw/assessments.csv', index=False)
print("✓ Saved: data/raw/assessments.csv")

exam_results.to_csv('data/raw/exam_results.csv', index=False)
print("✓ Saved: data/raw/exam_results.csv")

print("\n" + "="*60)
print("SAMPLE DATA CREATED SUCCESSFULLY!")
print("="*60)
print(f"Number of students: {n_students}")
print("\nFiles created in data/raw/:")
print("  - student_records.csv")
print("  - attendance.csv")
print("  - assessments.csv")
print("  - exam_results.csv")

# Display sample data
print("\n" + "="*60)
print("SAMPLE DATA PREVIEW")
print("="*60)

print("\nstudent_records.csv (first 5 rows):")
print(student_records.head())

print("\nattendance.csv (first 5 rows):")
print(attendance.head())

print("\nassessments.csv (first 5 rows):")
print(assessments.head())

print("\nexam_results.csv (first 5 rows):")
print(exam_results.head())

print("\n" + "="*60)
print("You can now run: python scripts/01_load_and_validate_data.py")
print("="*60)