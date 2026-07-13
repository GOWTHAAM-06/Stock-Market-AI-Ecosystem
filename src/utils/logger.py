import os
import logging
from logging.handlers import RotatingFileHandler

def get_logger(module_name: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        log_format = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = RotatingFileHandler(
            "logs/pipeline.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        
    return logger