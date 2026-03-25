from deepeval.models.base_model import DeepEvalBaseLLM
import asyncio
import os
from time import perf_counter
from typing import Dict, Tuple, Any

from langchain_google_vertexai import ChatVertexAI
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
# from langchain_community.chat_models import ChatOpenAI

from src.common.config.app_config import get_application_config
from src.common.service.logging.logger import error, info

CONFIG = get_application_config()


def _supports_json_schema(model_name: str) -> bool:
    """Returns True if the model supports OpenAI structured output (beta.chat.completions.parse).
    Gemini and other non-native OpenAI models routed through TFY do NOT support this endpoint.
    """
    m = model_name.lower()
    # Only native OpenAI and Azure OpenAI models support json_schema structured output
    native_openai = m.startswith("openai-main/") or m.startswith("openai/") or m.startswith("azure/")
    return native_openai


def _supports_reasoning_effort(model_name: str) -> bool:
    """Returns True if the model supports the reasoning_effort parameter."""
    m = model_name.lower()
    return any([
        # OpenAI reasoning models
        "o4-mini" in m, "o4-preview" in m, "o3-pro" in m, "o3-mini" in m, "/o3" in m,
        "/o1" in m, "o1-mini" in m, "o1-preview" in m,
        "gpt-5" in m, "codex-mini" in m,
        # Anthropic
        "claude-opus-4" in m, "claude-sonnet-4" in m, "claude-3-7" in m, "claude-sonnet-3.7" in m,
        # Gemini — 2.5 and 3 series support reasoning_effort (confirmed via CLI test)
        "gemini-2.5" in m, "gemini-3" in m,
        # Groq reasoning models
        "deepseek-r1" in m, "qwen3" in m, "qwen-3" in m, "gpt-oss" in m,
        # xAI
        "grok-3-mini" in m,
    ])


def _resolve_model_name(model_name: str | None = None) -> str:
    if model_name and model_name.strip():
        return model_name.strip()

    configured_model_name = CONFIG.get("llm.model_name")
    if isinstance(configured_model_name, str) and configured_model_name.strip():
        return configured_model_name.strip()

    env_model_name = os.getenv("PROMPT_TUNER_MODEL_NAME") or os.getenv("LLM_MODEL_NAME")
    if isinstance(env_model_name, str) and env_model_name.strip():
        return env_model_name.strip()

    return "openai-main/gpt-4o"


def get_recommendation_response_schema() -> dict[str, Any]:
    """JSON schema for recommendation endpoint output."""
    return {
        "type": "object",
        "properties": {
            "total_score": {"type": "integer"},
            "criteria_scores": {
                "type": "object",
                "properties": {
                    "clarity_and_specificity": {"type": "integer"},
                    "structure_and_organization": {"type": "integer"},
                    "output_specification": {"type": "integer"},
                    "contextual_guidance": {"type": "integer"},
                    "error_handling": {"type": "integer"},
                },
                "required": [
                    "clarity_and_specificity",
                    "structure_and_organization",
                    "output_specification",
                    "contextual_guidance",
                    "error_handling",
                ],
                "additionalProperties": False,
            },
            "explanations": {
                "type": "object",
                "properties": {
                    "clarity_and_specificity": {"type": "string"},
                    "structure_and_organization": {"type": "string"},
                    "output_specification": {"type": "string"},
                    "contextual_guidance": {"type": "string"},
                    "error_handling": {"type": "string"},
                },
                "required": [
                    "clarity_and_specificity",
                    "structure_and_organization",
                    "output_specification",
                    "contextual_guidance",
                    "error_handling",
                ],
                "additionalProperties": False,
            },
            "recommendations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["total_score", "criteria_scores", "explanations", "recommendations"],
        "additionalProperties": False,
    }


def get_truefoundry_llm(
    model_name: str | None = None,
    response_schema: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
):
    selected_model_name = _resolve_model_name(model_name)
    info(f"Using TrueFoundry model: {selected_model_name}")

    model_kwargs: dict[str, Any] = {
        "extra_headers": {
            "X-TFY-METADATA": "{}",
            "X-TFY-LOGGING-CONFIG": '{"enabled": true}',
        },
    }
    if response_schema:
        if _supports_json_schema(selected_model_name):
            model_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured-output",
                    "schema": response_schema,
                },
            }
            info(f"[LLM] Using json_schema structured output for model={selected_model_name}")
        else:
            # For Gemini and other non-native-OpenAI models, ANY response_format in model_kwargs
            # causes langchain_openai>=0.3 to route through beta.chat.completions.parse which
            # these models don't support. Skip response_format entirely — the prompt instructs
            # the model to return JSON and we parse the plain text response downstream.
            info(f"[LLM] Skipping response_format (unsupported via TFY) for model={selected_model_name}")
    resolved_temperature = temperature if temperature is not None else 0.1
    # Anthropic requires temperature=1 when extended thinking (reasoning_effort) is enabled
    if reasoning_effort and reasoning_effort != "none" and "claude" in selected_model_name.lower():
        resolved_temperature = 1.0

    kwargs: dict[str, Any] = {
        "model": selected_model_name,
        "temperature": resolved_temperature,
        "max_tokens": max_tokens or 15000,
        "streaming": False,
        "openai_api_key": os.getenv("TFY_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL"),
        "model_kwargs": model_kwargs,
    }
    if reasoning_effort and reasoning_effort != "none":
        if _supports_reasoning_effort(selected_model_name):
            kwargs["reasoning_effort"] = reasoning_effort
            info(f"[LLM] reasoning_effort={reasoning_effort} applied for model={selected_model_name}")
        else:
            info(f"[LLM] reasoning_effort={reasoning_effort} SKIPPED (not supported) for model={selected_model_name}")

    info(f"[LLM] ChatOpenAI kwargs: model={selected_model_name} | temperature={kwargs.get('temperature')} | max_tokens={kwargs.get('max_tokens')} | reasoning_effort={kwargs.get('reasoning_effort', 'none')}")
    return ChatOpenAI(**kwargs)


class TrueFoundryLLM(DeepEvalBaseLLM):
    """
    Custom TrueFoundry LLM adapter for DeepEval.
    
    This class adapts the TrueFoundry ChatOpenAI model to work with DeepEval's
    evaluation framework by implementing the required interface methods.
    """
    
    def __init__(self, model=None):
        self.model = model if model is not None else get_truefoundry_llm()
    
    def load_model(self):
        """Load the underlying model."""
        return self.model
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the model for the given prompt.
        
        Args:
            prompt: The input prompt text
            
        Returns:
            The generated response text
        """
        chat_model = self.load_model()
        # Handle both string prompts and message lists
        if isinstance(prompt, str):
            from langchain.schema import HumanMessage
            messages = [HumanMessage(content=prompt)]
        else:
            messages = prompt
        return chat_model.invoke(messages).content
    
    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate a response from the model.
        
        Args:
            prompt: The input prompt text
            
        Returns:
            The generated response text
        """
        chat_model = self.load_model()
        # Handle both string prompts and message lists
        if isinstance(prompt, str):
            from langchain.schema import HumanMessage
            messages = [HumanMessage(content=prompt)]
        else:
            messages = prompt
        res = await chat_model.ainvoke(messages)
        return res.content
    
    def get_model_name(self):
        return f"TrueFoundry {self.model.model_name}"

    def _make_json_model(self):
        """Return a copy of the model with json_object response format enforced."""
        base = self.model
        extra_headers = (base.model_kwargs or {}).get("extra_headers", {})
        json_kwargs = {
            "extra_headers": extra_headers,
            "response_format": {"type": "json_object"},
        }
        kwargs: dict[str, Any] = dict(
            model=base.model_name,
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            streaming=False,
            openai_api_key=base.openai_api_key,
            base_url=str(base.openai_api_base),
            model_kwargs=json_kwargs,
        )
        re = getattr(base, "reasoning_effort", None)
        if re and re != "none" and _supports_reasoning_effort(base.model_name):
            kwargs["reasoning_effort"] = re
        return ChatOpenAI(**kwargs)

    def generate_with_schema(self, prompt, schema=None, **kwargs):
        """Force json_object mode so the model returns clean JSON without markdown fences or reasoning prefixes."""
        chat_model = self._make_json_model()
        if isinstance(prompt, str):
            from langchain.schema import HumanMessage
            messages = [HumanMessage(content=prompt)]
        else:
            messages = prompt
        return chat_model.invoke(messages).content

    async def a_generate_with_schema(self, prompt, schema=None, **kwargs):
        chat_model = self._make_json_model()
        if isinstance(prompt, str):
            from langchain.schema import HumanMessage
            messages = [HumanMessage(content=prompt)]
        else:
            messages = prompt
        res = await chat_model.ainvoke(messages)
        return res.content


class AzureOpenAI(DeepEvalBaseLLM):
    """
    Custom Azure OpenAI model adapter for DeepEval.

    This class adapts the Azure OpenAI model to work with DeepEval's
    evaluation framework by implementing the required interface methods.
    """

    def __init__(self, model):
        """
        Initialize the adapter with an Azure OpenAI model.

        Args:
            model: An instance of AzureChatOpenAI to wrap
        """
        self.model = model

    def load_model(self):
        """Load the underlying model."""
        return self.model

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the model for the given prompt.

        Args:
            prompt: The input prompt text

        Returns:
            The generated response text
        """
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate a response from the model.

        Args:
            prompt: The input prompt text

        Returns:
            The generated response text
        """
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        """Return the name of the model."""
        return "Custom Azure OpenAI Model"
