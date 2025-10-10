"""
Deployment Utilities Package

Provides common utilities for NLP deployment:
- Logging configuration
- Environment management  
- Model caching
- Helper functions
"""

from .logging_config import setup_logging, get_logger, log_request, log_error
from .env_config import load_config, get_api_key, get_env_var
from .model_cache import ModelCache, get_model_cache

__all__ = [
    # Logging
    'setup_logging',
    'get_logger',
    'log_request',
    'log_error',

    # Environment
    'load_config',
    'get_api_key',
    'get_env_var',

    # Model Cache
    'ModelCache',
    'get_model_cache',
]

__version__ = '1.0.0'
