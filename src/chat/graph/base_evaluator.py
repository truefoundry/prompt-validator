import requests
import json
import asyncio
from abc import ABC, abstractmethod
from truefoundry.ml import get_client

from src.common.config.app_config import get_application_config
from src.common.service.llm_prompt.prompt_service import PromptService
from src.common.service.logging.logger import error, info

class PromptEvaluator(ABC):
    """Base class for prompt evaluation strategies."""
    
    def __init__(
        self,
        model_name: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ):
        self.config = get_application_config()
        self.tf_client = get_client()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
    
    async def get_all_tests_db(self, prompt_fqn):
        """Fetch test cases from the truefoundry artifact."""

        try:
            prompt_name = prompt_fqn.split("/")[-1].split(":")[0]
            ml_repo = self.config.get('application_details.ml_repo')
            artifact_version = self.tf_client.get_artifact_version(ml_repo=ml_repo, name=f"{prompt_name}_tests")
            info(f"Version: {artifact_version}")
            # download it to disk
            # `download_path` points to a directory that has all contents of the artifact
            download_path = artifact_version.download(path="src/chat/data/test_cases/", overwrite=True)
            with open(f"{download_path}/tests.json", "r") as f:
                all_tests = json.load(f)
            return all_tests
        except requests.exceptions.RequestException as e:
            error(f"Error fetching test cases: {str(e)}")
            return []
    
    async def get_test_responses(self, prompt_fqn, all_tests, *, system_prompt=None, user_prompt_template=None):
        """Get responses for all test cases in parallel.

        When *system_prompt* is provided the LLM is invoked directly from the
        raw text instead of resolving a TFY FQN.
        """
        async def get_single_response(test):
            if system_prompt:
                response = await PromptService.get_prompt_response_from_text(
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    data=test["data"],
                    model_name=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                )
            else:
                response = await PromptService.get_prompt_response(
                    prompt_fqn=prompt_fqn,
                    data=test["data"],
                    model_name=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                )
            test["actual_output"] = response
            return test

        tasks = [get_single_response(test) for test in all_tests]
        all_tests = await asyncio.gather(*tasks)
        return all_tests
    
    @abstractmethod
    async def evaluate_tests(self, all_tests, is_rag=False):
        """Evaluate the test cases using the specific evaluation strategy."""
        pass
    
    @abstractmethod
    def get_final_report(self, all_tests, evaluation_results=None):
        """Generate the final evaluation report."""
        pass
    
    async def verify_tests(self, state):
        """Main method to verify tests using the specific evaluation strategy."""
        prompt_fqn = state['request'].get('prompt_fqn')
        system_prompt = state['request'].get('system_prompt')
        user_prompt_template = state['request'].get('user_prompt_template')

        provided_tests = state['request'].get('test_cases')
        if provided_tests:
            all_tests = provided_tests
            info(f"Using {len(all_tests)} test cases from request payload.")
        elif prompt_fqn:
            all_tests = await self.get_all_tests_db(prompt_fqn)
            all_tests = all_tests[:10]
        else:
            error("No test cases provided and no prompt FQN to fetch from artifacts.")
            return []
        info(f"Number of test cases created: {len(all_tests)}")
        
        all_tests = await self.get_test_responses(
            prompt_fqn, all_tests,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )
        info("Test responses received.")
        
        evaluation_results = await self.evaluate_tests(all_tests, state['request']['is_rag'])
        info("Test evaluation completed.")
        report = self.get_final_report(all_tests, evaluation_results)
        info("Final report generated.")

        return report