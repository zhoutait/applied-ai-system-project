"""
Logger Configuration — StudyMind AI.

Sets up structured logging to both console and a rotating log file.
"""

import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> None:
    """
    Configure application-wide logging.

    Creates a 'logs/' directory and sets up:
      - Console handler (INFO level, coloured prefix)
      - Rotating file handler (DEBUG level, full details)

    Parameters
    ----------
    log_dir : str
        Directory where log files will be stored.
    log_level : int
        Minimum log level for the console handler.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, "studymind.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates on re-initialisation
    root_logger.handlers.clear()

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)

    # --- Rotating file handler ---
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logging.info("Logging initialised. Log file: %s", log_file)
