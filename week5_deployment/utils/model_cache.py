"""
Model Cache Manager
Manage model loading, caching, and lifecycle.
"""

import logging
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class ModelCache:
    """Cache for transformer models to avoid reloading."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._pipelines: Dict[str, Any] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ModelCache initialized on device: {self.device}")

    def get_pipeline(self, task: str, model_name: Optional[str] = None, **kwargs) -> Any:
        """Get or create a Hugging Face pipeline."""
        cache_key = f"{task}_{model_name or 'default'}"

        if cache_key not in self._pipelines:
            logger.info(f"Creating pipeline: {task}")
            pipeline_kwargs = {
                'device': 0 if self.device == "cuda" else -1,
                **kwargs
            }

            if model_name:
                pipeline_kwargs['model'] = model_name

            self._pipelines[cache_key] = pipeline(task, **pipeline_kwargs)
            logger.info(f"Pipeline created: {task}")

        return self._pipelines[cache_key]

    def clear_cache(self):
        """Clear all cached models."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self._pipelines.clear()


_global_cache: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Get global model cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ModelCache()
    return _global_cache
