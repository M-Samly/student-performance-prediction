### run_all.py
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_script(script_path, description):
    """Run a script and handle errors"""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print(f"Script: {script_path}")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✓ COMPLETED: {description} (took {duration:.2f} seconds)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ FAILED: {description}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {description}")
        print(f"Error: {e}")
        return False

def main():
    """Run all scripts in sequence"""
    
    print("\n" + "="*70)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("COMPLETE PIPELINE EXECUTION")
    print("="*70)
    
    overall_start = datetime.now()
    
    scripts = [
        ("scripts/01_load_and_validate_data.py", "Step 1: Load and Validate Data"),
        ("scripts/02_preprocess_data.py", "Step 2: Preprocess Data"),
        ("scripts/03_train_baseline.py", "Step 3: Train Baseline Model"),
        ("scripts/04_train_improved_models.py", "Step 4: Train Improved Models"),
        ("scripts/05_evaluate_models.py", "Step 5: Evaluate Models"),
        ("scripts/06_generate_predictions.py", "Step 6: Generate Predictions"),
        ("scripts/07_generate_analytics.py", "Step 7: Generate Analytics"),
        ("scripts/08_create_risk_report.py", "Step 8: Create Risk Report"),
    ]
    
    results = []
    
    for script_path, description in scripts:
        if not Path(script_path).exists():
            print(f"\n✗ SKIPPED: {description}")
            print(f"   Script not found: {script_path}")
            results.append(False)
            continue
        
        success = run_script(script_path, description)
        results.append(success)
        
        if not success:
            print("\n" + "="*70)
            print("PIPELINE STOPPED DUE TO ERROR")
            print("="*70)
            sys.exit(1)
    
    overall_end = datetime.now()
    total_duration = (overall_end - overall_start).total_seconds()
    
    # Summary
    print("\n\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    
    for i, (script_path, description) in enumerate(scripts):
        status = "✓ PASSED" if results[i] else "✗ FAILED"
        print(f"{status}: {description}")
    
    successful = sum(results)
    total = len(results)
    
    print(f"\nTotal: {successful}/{total} steps completed successfully")
    print(f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    print("\n" + "="*70)
    print("OUTPUTS LOCATION:")
    print("="*70)
    print("Reports: outputs/reports/")
    print("Figures: outputs/figures/")
    print("Models: models/")
    print("Logs: outputs/logs/")
    
    if all(results):
        print("\n✓✓✓ ALL STEPS COMPLETED SUCCESSFULLY ✓✓✓\n")
        return 0
    else:
        print("\n✗✗✗ SOME STEPS FAILED ✗✗✗\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())