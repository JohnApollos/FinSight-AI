import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from backend.app.config import settings

def setup_logger(name: str) -> logging.Logger:
    """Configures a standardized, production-ready console & file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL)
    
    # Avoid duplicate handlers if setup multiple times
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (standard stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Rotating file handler (5MB limit per file, rolling cache of 3 files)
    try:
        log_dir = "backend/data/logs"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "finlens_api.log"),
            maxBytes=5*1024*1024, # 5MB
            backupCount=3,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not configure rolling file logger: {e}")
        
    return logger
