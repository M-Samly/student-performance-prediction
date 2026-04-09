import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.data_validator import DataValidator
from src.config import PROCESSED_DATA_DIR, TARGET_COL
from src.utils.file_handler import save_csv, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Load and validate all OULAD data files"""
    
    logger.info("="*60)
    logger.info("STEP 1: DATA LOADING AND VALIDATION")
    logger.info("="*60)
    
    # Load data
    loader = DataLoader()
    data_files = loader.load_all_files()
    
    # Display basic info about each file
    for name, df in data_files.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Rows: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")
    
    # Merge datasets
    logger.info("\nMerging datasets...")
    merged_df = loader.merge_datasets()
    
    # Get data summary
    summary = loader.get_data_summary()
    logger.info(f"\nMerged dataset summary:")
    logger.info(f"  Total rows: {summary['total_rows']}")
    logger.info(f"  Total columns: {summary['total_columns']}")
    logger.info(f"  Unique students: {summary['unique_students']}")
    logger.info(f"  Unique modules: {summary['unique_modules']}")
    logger.info(f"  Target distribution: {summary['target_distribution']}")
    
    # Validate data
    logger.info("\nValidating data...")
    validator = DataValidator(merged_df)
    validation_report = validator.validate_all(target_col=TARGET_COL)
    
    # Save merged data
    output_path = PROCESSED_DATA_DIR / "merged_raw_data.csv"
    save_csv(merged_df, output_path)
    
    # Save validation report
    report_path = PROCESSED_DATA_DIR / "validation_report.json"
    save_json(validation_report, report_path)
    
    logger.info(f"\nMerged data saved to: {output_path}")
    logger.info(f"Validation report saved to: {report_path}")
    logger.info("\n" + "="*60)
    logger.info("DATA LOADING AND VALIDATION COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    main()