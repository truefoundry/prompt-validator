from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from src.common.config.app_config import get_application_config

CONFIG = get_application_config()


# Update Dialog Stack
def update_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


def _get_interrupt_states() -> list[str]:
    """Dynamically get all possible interrupt states from assistant configurations."""
    interrupt_states = []
    assistants_config = CONFIG.get("assistants.definitions", {})

    for assistant_name, assistant_config in assistants_config.items():
        if assistant_config.get("sensitive_tools"):
            # Use the dictionary key as the assistant name
            interrupt_state = f"{assistant_name}_action_sensitive_tools"
            interrupt_states.append(interrupt_state)
    return interrupt_states


def _get_assistant_names() -> list[str]:
    """Dynamically get all assistant names from configuration."""
    return list(CONFIG.get("assistants.definitions", {}).keys())


# Create type variables for the Literals
# Note: str is used as a fallback type if the dynamic values can't be determined
InterruptStateLiteral = str
AssistantNameLiteral = str


# Agent State Model
# Annoted literals cant be filled at runtime.
# So we use str as a fallback type.
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    request: Optional[dict]
    interrupt_question: Optional[str]
    interrupt_state: Annotated[
        list[InterruptStateLiteral],
        update_stack,
    ]

    dialog_state: Annotated[
        list[AssistantNameLiteral],
        update_stack,
    ]


# Runtime validation can be done separately in base_assistant_util.py
_valid_interrupt_states = set(_get_interrupt_states())
_valid_assistant_names = set(_get_assistant_names())


def validate_interrupt_state(state: str) -> bool:
    """Validate that an interrupt state is valid."""
    return state in _valid_interrupt_states


def validate_assistant_name(name: str) -> bool:
    """Validate that an assistant name is valid."""
    return name in _valid_assistant_names
