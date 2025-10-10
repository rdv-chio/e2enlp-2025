"""
Environment Configuration
Load and validate environment variables.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


def load_env_file(env_file: str = ".env") -> bool:
    """Load environment variables from .env file."""
    current_dir = Path.cwd()

    for _ in range(3):
        env_path = current_dir / env_file
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")
            return True
        current_dir = current_dir.parent

    logger.warning(f"No {env_file} file found")
    return False


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get environment variable with validation."""
    value = os.getenv(key, default)

    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' not found")

    return value


def get_api_key(service: str) -> str:
    """Get API key for a service."""
    key = get_env_var(service, required=True)

    if not key or key.startswith('your-') or key.startswith('sk-your'):
        raise ValueError(f"Invalid {service}. Please set it in .env file.")

    return key


def load_config() -> Dict[str, Any]:
    """Load complete application configuration."""
    load_env_file()

    return {
        'openai_api_key': get_env_var('OPENAI_API_KEY'),
        'anthropic_api_key': get_env_var('ANTHROPIC_API_KEY'),
        'api_host': get_env_var('API_HOST', '0.0.0.0'),
        'api_port': int(get_env_var('API_PORT', '8000')),
        'log_level': get_env_var('LOG_LEVEL', 'INFO'),
        'environment': get_env_var('ENVIRONMENT', 'development'),
    }
