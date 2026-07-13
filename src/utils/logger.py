import os
import logging
from logging.handlers import RotatingFileHandler

def get_logger(module_name: str) -> logging.Logger:
    """
    Creates a production-ready logger that outputs both to the 
    console and a rolling log file.
    """
    # Ensure a logs directory exists at the root
    os.makedirs("logs", exist_ok=True)
    
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers if logger is called multiple times
    if not logger.handlers:
        # Define standard log format: Timestamp | Level | Module | Message
        log_format = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 1. File Handler: Keeps last 5MB of logs, rotates automatically
        file_handler = RotatingFileHandler(
            "logs/pipeline.log", maxBytes=5*1024*1024, backupCount=3
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        
        # 2. Console Handler: Shows live logs in the VS Code terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        
    return logger