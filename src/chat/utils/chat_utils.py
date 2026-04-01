from typing import Callable, List

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from src.chat.models.state_model import State
from src.common.config.app_config import get_application_config
from src.common.models.prompt_model import PromptDetail
from src.common.service.llm_prompt.prompt_service import PromptService
from src.common.service.logging.logger import error

config = get_application_config()


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    """Create entry node - remains synchronous as it's a factory function."""

    async def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the prescriptions, stores other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the complete_or_escalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


def print_event(event: dict, _printed: set, max_length=1500):
    """Print event - remains synchronous as it's just printing."""
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


# This node will be shared for exiting all specialized assistants
async def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


async def get_prompt_details(prompt_ids: list) -> List[PromptDetail]:
    """Get prompt details asynchronously."""
    return await PromptService.get_prompt_details(prompt_ids)

