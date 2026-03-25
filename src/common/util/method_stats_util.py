import functools
import time
from typing import Callable, Any

from src.common.service.logging.logger import (
    add_method_execution_stats,
    add_functional_tags,
    add_functional_metrics,
    add_llm_execution_stats,
    signal,
    error
)



def method_exec_stats(func: Callable) -> Callable:
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        return _execute_with_stats(func, *args, **kwargs)

    return sync_wrapper


def _execute_with_stats(func: Callable, *args, **kwargs) -> Any:
    start_time = time.perf_counter()
    signal(f"Executing {func.__name__} method")

    try:
        result = func(*args, **kwargs)

    except Exception as e:
        error(f"Error executing {func.__name__} method:", e)
        return None

    signal(f"Executed {func.__name__} method")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    stats = {func.__name__: round(elapsed_time, 3)}
    add_method_execution_stats(stats)
    return result
