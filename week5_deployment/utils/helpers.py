"""
Helper Functions
Common utility functions for deployment.
"""

from typing import Any, Dict, Optional


def format_response(data: Any, success: bool = True, message: Optional[str] = None) -> Dict:
    """Format API response in consistent structure."""
    response = {
        'success': success,
        'data': data
    }

    if message:
        response['message'] = message

    return response


def validate_text(text: str, min_length: int = 1, max_length: int = 10000) -> tuple:
    """Validate text input."""
    if not text or not text.strip():
        return False, "Text cannot be empty"

    if len(text) < min_length:
        return False, f"Text too short (minimum {min_length} characters)"

    if len(text) > max_length:
        return False, f"Text too long (maximum {max_length} characters)"

    return True, None
