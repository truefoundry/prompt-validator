import logging
import traceback
from typing import Optional, Dict, Any

# Configure logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)

# Add console handler if no handlers are present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Global storage for stats and metrics
_method_stats: Dict[str, Any] = {}
_functional_tags: Dict[str, Any] = {}
_functional_metrics: Dict[str, Any] = {}
_llm_stats: Dict[str, Any] = {}

def add_method_execution_stats(method_stats: Dict[str, Any]) -> None:
    """Add method execution statistics.
    
    Args:
        method_stats (Dict[str, Any]): Dictionary of method execution statistics
    """
    if not isinstance(method_stats, dict):
        raise ValueError("method_stats must be a dictionary")
    
    for key, value in method_stats.items():
        new_key = key
        counter = 1
        while new_key in _method_stats:
            counter += 1
            new_key = f"{key}_{counter}"
        _method_stats[new_key] = value

def add_functional_tags(tags: Dict[str, Any]) -> None:
    """Add functional tags.
    
    Args:
        tags (Dict[str, Any]): Dictionary of functional tags
    """
    if not isinstance(tags, dict):
        raise ValueError("tags must be a dictionary")
    _functional_tags.update(tags)

def add_functional_metrics(metrics: Dict[str, Any]) -> None:
    """Add functional metrics.
    
    Args:
        metrics (Dict[str, Any]): Dictionary of functional metrics
    """
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary")
    _functional_metrics.update(metrics)

def add_llm_execution_stats(llm_stats: Dict[str, Any]) -> None:
    """Add LLM execution statistics.
    
    Args:
        llm_stats (Dict[str, Any]): Dictionary of LLM execution statistics
    """
    if not isinstance(llm_stats, dict):
        raise ValueError("llm_stats must be a dictionary")
    _llm_stats.update(llm_stats)

def _get_stats_context() -> Dict[str, Any]:
    """Get all stats and metrics as a single context dictionary."""
    return {
        "method_stats": _method_stats,
        "functional_tags": _functional_tags,
        "functional_metrics": _functional_metrics,
        "llm_stats": _llm_stats
    }

def info(message: str) -> None:
    """Log an informational message.
    
    Args:
        message (str): The message to log
    """
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    context = _get_stats_context()
    logger.info(message, extra=context)

def error(message: str, exception: Optional[Exception] = None) -> None:
    """Log an error message with optional exception details.
    
    Args:
        message (str): The error message to log
        exception (Exception, optional): The exception that occurred
    """
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    
    context = _get_stats_context()
    if exception:
        logger.error(f"{message}\nException: {type(exception).__name__}: {str(exception)}\n{traceback.format_exc()}", extra=context)
    else:
        logger.error(message, extra=context)

def signal(message: str) -> None:
    """Log a signal/event message.
    
    Args:
        message (str): The signal message to log
    """
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    logger.info(f"SIGNAL: {message}") 