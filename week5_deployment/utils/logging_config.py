"""
Logging Configuration
Centralized logging setup for all deployment services.
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def get_logger(name):
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_request(request, response_time=None):
    """Log HTTP request details."""
    logger = get_logger('api.requests')
    log_message = f"{request.method} {request.url.path}"
    if response_time:
        log_message += f" - {response_time:.4f}s"
    logger.info(log_message)


def log_error(error, context=None):
    """Log error with context."""
    logger = get_logger('api.errors')
    error_message = f"{type(error).__name__}: {str(error)}"
    if context:
        error_message += f" | Context: {context}"
    logger.error(error_message, exc_info=True)
