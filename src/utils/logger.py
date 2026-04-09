### src/utils/logger.py
import logging
import sys
from pathlib import Path
from datetime import datetime
from src.config import LOGS_DIR

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Log file name (optional)
        level: Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str):
    """Get or create logger"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{name}_{timestamp}.log"
    return setup_logger(name, log_file)