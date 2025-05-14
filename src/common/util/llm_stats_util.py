import asyncio
import functools
import time
from typing import Callable, Any

from langchain_core.messages import BaseMessage, AIMessage

from src.common.models.llm_stats_model import LLMStats
from src.common.service.logging.logger import (
    add_llm_execution_stats,
    add_functional_tags,
    add_functional_metrics,
    signal
)


def llm_exec_stats(func: Callable) -> Callable:
    """
    Decorator to log the execution time of a llm.
    Args:
        func: The function to be decorated.
    Returns:
        The decorated function.
    """

    # Async Processing
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        return await _execute_with_logging(func, *args, **kwargs)

    # Sync Processing
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        return _execute_with_logging(func, *args, **kwargs)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


async def _execute_with_logging(func: Callable, *args, **kwargs) -> Any:
    """
    Log the execution time of the function.
    Args:
        func: The function to be executed.
        args: The arguments to the function.
        kwargs: The keyword arguments to the function.
    Returns:
        The result of the function.
    """
    start_time = time.perf_counter()
    result = (
        await func(*args, **kwargs)
        if asyncio.iscoroutinefunction(func)
        else func(*args, **kwargs)
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    _log_llm_stats(result, func.__name__, start_time, end_time, elapsed_time, args)
    return result


def _log_llm_stats(
    result: Any,
    func_name: str,
    start_time: float,
    end_time: float,
    elapsed_time: float,
    args: tuple,
) -> None:
    """
    Log the LLM stats.
    Args:
        result: The result of the function.
        func_name: The name of the function.
        start_time: The start time of the function.
        end_time: The end time of the function.
        elapsed_time: The elapsed time of the function.
        args: The arguments to the function.
    Returns:
        None
    """

    # Collecting LLM stats is Only applicable for AIMessage objects
    # Use the args for determining the selected tools and selected prompt
    if not isinstance(result, dict) or "messages" not in result or args is None:
        signal(f"Skipping LLM stats for {func_name} as no messages found")
        return

    messages = result["messages"]
    base_messages = list(filter(lambda msg: isinstance(msg, BaseMessage), messages))

    if not base_messages:
        signal(f"Skipping LLM stats for {func_name} as no base messages found")
        return

    state = args[1] if args else {}
    selected_tools = state.get("selected_tools", [])
    selected_prompt = state.get("selected_completion_prompt", {})

    for msg in base_messages:
        if isinstance(msg, AIMessage):
            # Check if the message has tool calls and usage metadata. This indicates the message is a tool response
            if (
                hasattr(msg, "tool_calls")
                and len(msg.tool_calls) > 0
                and hasattr(msg, "usage_metadata")
                and msg.usage_metadata
            ):
                _log_tool_usage_stats(
                    msg, func_name, start_time, end_time, elapsed_time, selected_tools
                )

            # Check if the message has usage metadata and content. This indicates the message is a prompt response
            elif (
                hasattr(msg, "usage_metadata")
                and msg.usage_metadata
                and hasattr(msg, "content")
                and msg.content
            ):
                _log_prompt_usage_stats(
                    msg, func_name, start_time, end_time, elapsed_time, selected_prompt
                )


def _log_tool_usage_stats(
    msg: AIMessage,
    func_name: str,
    start_time: float,
    end_time: float,
    elapsed_time: float,
    selected_tools: list,
) -> None:
    """
    Log the tool usage stats. This function is called when the message is a tool response.
    The function definition is available in the selected tools.
    Example - promptFQN, promptDescription, prompt, functionParameters, functionProvider, functionModel.
    The AI result comprises the LLM usage metadata and tool calls.
    Example - input_tokens, output_tokens, total_tokens, name, args, id, type.
    Merge the selected tool and AI result to build the LLM stats object.
    Args:
        msg: The AIMessage object.
        func_name: The name of the function.
        start_time: The start time of the function.
        end_time: The end time of the function.
        elapsed_time: The elapsed time of the function.
        selected_tools: The selected tools.
    Returns:
        None
    """
    tool_calls = getattr(msg, "tool_calls", [])
    usage_metadata = getattr(msg, "usage_metadata", {})

    if not tool_calls or not usage_metadata:
        signal(f"Skipping LLM stats for {func_name} as no tool calls found")
        return

    input_tokens = usage_metadata.get("input_tokens", 0)
    output_tokens = usage_metadata.get("output_tokens", 0)
    total_tokens = usage_metadata.get("total_tokens", 0)

    for tool_call in tool_calls:
        # The Selected tool comprises the requested Function Definition from X42
        selected_tool = next(
            (
                tool
                for tool in selected_tools
                if tool["functionName"] == tool_call["name"]
            ),
            {},
        )

        # Build the LLM stats object
        llm_stats: LLMStats = LLMStats(
            node_name=func_name,
            node_start_time=start_time,
            node_end_time=end_time,
            response_time=elapsed_time,
            tool_name=tool_call.get("name", ""),
            tool_args=tool_call.get("args", {}),
            tool_id=tool_call.get("id", ""),
            tool_type=tool_call.get("type", ""),
            input_token_size=input_tokens,
            output_token_size=output_tokens,
            total_tokens=total_tokens,
            prompt_fqn=selected_tool.get("functionId", ""),
            prompt_short_description=selected_tool.get("functionDescription", ""),
            generated_prompt=selected_tool.get("functionPrompt", ""),
            additional_data=None,
            prompt_parameters=selected_tool.get("functionParameters", {}),
            provider=selected_tool.get("functionProvider", ""),
            model=selected_tool.get("functionModel", ""),
            rag_enabled=False,
            rag_datastore=False,
            answer=tool_call,
        )

        add_llm_execution_stats({llm_stats.prompt_fqn: llm_stats})


def _log_prompt_usage_stats(
    msg: AIMessage,
    func_name: str,
    start_time: float,
    end_time: float,
    elapsed_time: float,
    selected_prompt: dict,
) -> None:
    """
    Log the usage statistics for the prompt.
    The prompt response includes the LLM usage metadata and content, such as input tokens, output tokens, total tokens,
    and content.
    The selected prompt contains the requested prompt definition from X42, including prompt ID, prompt description,
    prompt, parameters, provider, and model.
    Combine the selected prompt and AI result to create the LLM stats object.
    Args:
        msg: The AIMessage object.
        func_name: The name of the function.
        start_time: The start time of the function.
        end_time: The end time of the function.
        elapsed_time: The elapsed time of the function.
        selected_prompt: The selected prompt.
    Returns:
        None
    """
    usage_metadata = getattr(msg, "usage_metadata", {})

    if not usage_metadata:
        signal(f"Skipping LLM stats for {func_name} as no usage metadata found")
        return

    input_tokens = usage_metadata.get("input_tokens", 0)
    output_tokens = usage_metadata.get("output_tokens", 0)
    total_tokens = usage_metadata.get("total_tokens", 0)

    # Build the LLM stats object
    llm_stats: LLMStats = LLMStats(
        node_name=func_name,
        node_start_time=start_time,
        node_end_time=end_time,
        response_time=elapsed_time,
        input_token_size=input_tokens,
        output_token_size=output_tokens,
        total_tokens=total_tokens,
        prompt_fqn=selected_prompt.get("promptFQN", ""),
        prompt_short_description=selected_prompt.get("promptDescription", ""),
        generated_prompt=selected_prompt.get("prompt", ""),
        additional_data=None,
        prompt_parameters=selected_prompt.get("parameters", {}),
        provider=selected_prompt.get("provider", ""),
        model=selected_prompt.get("model", ""),
        rag_enabled=False,
        rag_datastore=False,
        answer=msg.content,
    )

    add_llm_execution_stats({llm_stats.prompt_fqn: llm_stats})
