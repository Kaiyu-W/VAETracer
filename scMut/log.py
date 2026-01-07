import os
import logging
from .typing import Optional

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

def setup_logging(
    logger_name: str = "default_logger",
    log_file: Optional[str] = None,
    verbose: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration for a named logger with optional console and file output.

    Args:
        logger_name: Name of the logger.
        log_file: Path to log file; if None, no file handler is added.
        verbose: Whether to enable console output.

    Returns:
        Configured logger instance.
    """

    # Create a named logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Log all levels

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.propagate = False
    
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def _check_dir_exist(file: str) -> None:
    """
    Check that the parent directory of the given file path exists.

    Raises ValueError if it does not.

    Args:
        file: Full path to a file.

    Raises:
        ValueError: If parent directory does not exist.
    """

    _dir = os.path.dirname(file)
    if _dir and not os.path.exists(_dir):
        raise ValueError(f"No dir {_dir}")

def add_file_handler(
    logger: logging.Logger,
    log_file: str,
) -> logging.Handler:
    """
    Add a file handler to an existing logger.

    Ensures directory exists, then attaches a FileHandler for debug-level logging.

    Args:
        logger: Target logger.
        log_file: Path to write logs; parent dir must exist.

    Returns:
        The newly created file handler.
    """

    _check_dir_exist(log_file)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return file_handler

def remove_file_handler(
    logger: logging.Logger,
    file_handler: logging.Handler,
) -> None:
    """
    Safely remove and close a file handler from a logger.

    Args:
        logger: Logger instance.
        file_handler: Handler to remove.
    """

    if file_handler in logger.handlers:
        file_handler.flush()
        file_handler.close()
        logger.removeHandler(file_handler)

def cleanup_logging(logger: logging.Logger) -> None:
    """
    Clean up all handlers associated with a logger.

    Flushes, closes, and removes every handler to prevent resource leaks.

    Args:
        logger: Logger to clean.
    """

    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

# Global default logger instance
logger = setup_logging()
