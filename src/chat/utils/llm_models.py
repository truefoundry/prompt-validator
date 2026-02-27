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
        model_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured-output",
                "schema": response_schema,
            },
        }

    return ChatOpenAI(
        model=selected_model_name,
        temperature=0.1,
        max_tokens=15000,
        streaming=False,
        openai_api_key=os.getenv("TFY_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        model_kwargs=model_kwargs,
    )


class TrueFoundryLLM(DeepEvalBaseLLM):
    """
    Custom TrueFoundry LLM adapter for DeepEval.
    
    This class adapts the TrueFoundry ChatOpenAI model to work with DeepEval's
    evaluation framework by implementing the required interface methods.
    """
    
    def __init__(self, model=None):
        """
        Initialize the adapter with a TrueFoundry model.
        
        Args:
            model: An instance of ChatOpenAI (TrueFoundry) to wrap. 
                   If None, will use get_truefoundry_llm()
        """
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
        """Return the name of the model."""
        return "TrueFoundry GPT-4o"


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
