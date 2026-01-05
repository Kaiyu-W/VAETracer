import os
import logging
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

def setup_logging(logger_name="default_logger", log_file=None, verbose=True):
    """
    Set up logging configuration for a named logger.
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

def _check_dir_exist(file):
    _dir = os.path.dirname(file)
    if _dir and not os.path.exists(_dir):
        raise ValueError(f"No dir {_dir}")

def add_file_handler(logger, log_file):
    _check_dir_exist(log_file)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return file_handler

def remove_file_handler(logger, file_handler):
    if file_handler in logger.handlers:
        file_handler.flush()
        file_handler.close()
        logger.removeHandler(file_handler)

def cleanup_logging(logger):
    """
    Clean up handlers for a specific named logger.
    """
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

logger = setup_logging()
