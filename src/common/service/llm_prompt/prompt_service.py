import asyncio
from typing import Dict, Any, Optional, List

from fastapi import HTTPException
from pydantic import ValidationError
from truefoundry import client, render_prompt
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage

from src.common.config.app_config import get_application_config
from src.common.models.prompt_model import PromptDetail
from src.common.service.logging.logger import error, info
from src.chat.utils.llm_models import get_truefoundry_llm

config = get_application_config()

class PromptService:
    @staticmethod
    async def get_prompt_details(
        prompt_fqn_list: List[str],
        request_headers: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Fetches prompt details using TrueFoundry client.

        Args:
            prompt_fqn_list: Prompt FQN list.
            request_headers: Optional request headers.

        Returns:
            List[PromptDetail]: The prompt details.

        Raises:
            HTTPException: If the prompt FQN is missing or if fetching fails.
        """
        if not prompt_fqn_list:
            raise HTTPException(status_code=400, detail="prompt_fqn_list is required")

        try:
            prompt_details = []
            for fqn in prompt_fqn_list:
                prompt_version_response = client.prompt_versions.get_by_fqn(fqn=fqn)
                prompt_template = prompt_version_response.data
                prompt_details.append(prompt_template.__dict__)
            return prompt_details

        except Exception as e:
            error(f"Error fetching prompt details: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch prompt details: {str(e)}"
            ) from e

    @staticmethod
    async def get_prompt_response(
        prompt_fqn: str,
        data: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Fetches LLM response for a given prompt and input using TrueFoundry client and LLM.

        Args:
            prompt_fqn: Prompt FQN.
            data: Input data to be processed by the prompt.
        Returns:
            Dict[str, Any]: The LLM response.

        Raises:
            HTTPException: If the prompt FQN is missing or if the API call fails.
        """
        if not prompt_fqn:
            raise HTTPException(status_code=400, detail="prompt_fqn is required")

        try:
            # Get prompt template
            prompt_version_response = client.prompt_versions.get_by_fqn(fqn=prompt_fqn)
            prompt_template = prompt_version_response.data.manifest

            # Render prompt with variables
            rendered_prompt = render_prompt(prompt_template, variables=data)
            
            # Convert messages to LangChain message types
            messages: List[BaseMessage] = []
            for msg in rendered_prompt['messages']:
                if msg['role'] == 'system':
                    messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))
            # Get LLM and invoke
            llm = get_truefoundry_llm()
            response = await llm.ainvoke(messages)
            return response.content

        except Exception as e:
            error(f"Error getting prompt response: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get prompt response: {str(e)}"
            ) from e
