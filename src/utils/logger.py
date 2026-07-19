"""
Logger Configuration
"""

import logging
from pathlib import Path


def setup_logger():

    log_dir = Path("logs")

    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("PredictiveMaintenance")

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(
            log_dir / "project.log"
        )

        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()

        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger