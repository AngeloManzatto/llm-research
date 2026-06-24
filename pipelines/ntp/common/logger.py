"""
Created on Sat Jun 21 21:20:07 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

###############################################################################
# Init Logger
###############################################################################
def setup_logger(pipeline_name, 
                 logs_base_dir="logs", 
                 log_level="DEBUG", 
                 log_to_stdout=True):
    """
    Initializes a structured loguru logger for a given pipeline.

    Args:
        pipeline_name (str): Name of the pipeline (e.g., 'wikipedia', 'commoncrawl').
        logs_base_dir (str): Base directory where logs are stored.
        log_level (str): Minimum log level to capture (DEBUG, INFO, etc.).
    """
    # Ensure log path exists
    logger.remove()

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    log_path = Path(logs_base_dir) / pipeline_name
    log_path.mkdir(parents=True, exist_ok=True)
    file_path = log_path / f"{pipeline_name}-{timestamp}.log.jsonl"

    logger.add(
        str(file_path),
        level=log_level,
        format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {message}",
        serialize=True,
        rotation="100 MB",
        compression="zip"
    )

    if log_to_stdout:
        logger.add(
            sink=sys.stdout,
            level=log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
        )

    logger.info(f"Logger initialized for pipeline: {pipeline_name}")
    
    return logger

