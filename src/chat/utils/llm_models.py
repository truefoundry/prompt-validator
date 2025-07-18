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


def get_truefoundry_llm():
    return ChatOpenAI(
        model="openai-main/gpt-4o",
        temperature=0.1,
        max_tokens=2500,
        streaming=False,
        openai_api_key=os.getenv("TFY_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

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
