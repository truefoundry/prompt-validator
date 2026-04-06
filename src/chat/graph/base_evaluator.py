import requests
import json
import asyncio
import time
import traceback
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
            info(f"[BaseEvaluator] Fetching test artifact | prompt={prompt_name} | ml_repo={ml_repo}")
            artifact_version = self.tf_client.get_artifact_version(ml_repo=ml_repo, name=f"{prompt_name}_tests")
            info(f"[BaseEvaluator] Artifact version fetched | version={artifact_version}")
            download_path = artifact_version.download(path="src/chat/data/test_cases/", overwrite=True)
            with open(f"{download_path}/tests.json", "r") as f:
                all_tests = json.load(f)
            info(f"[BaseEvaluator] Loaded {len(all_tests)} test cases from artifact")
            return all_tests
        except requests.exceptions.RequestException as e:
            error(f"[BaseEvaluator] Network error fetching test cases | {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return []
        except Exception as e:
            error(f"[BaseEvaluator] Error fetching test cases | {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return []
    
    async def get_test_responses(self, prompt_fqn, all_tests, *, system_prompt=None, user_prompt_template=None):
        """Get responses for all test cases in parallel."""
        mode = "raw_text" if system_prompt else "fqn"
        info(f"[BaseEvaluator] get_test_responses | mode={mode} | tests={len(all_tests)} | model={self.model_name}")
        t0 = time.time()

        async def get_single_response(test):
            tid = test.get("test_case_id", "?")
            t_start = time.time()
            try:
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
            except Exception as e:
                error(f"[BaseEvaluator] Failed to get response | test_id={tid} | {type(e).__name__}: {e}")
                test["actual_output"] = None
            test["latency_s"] = round(time.time() - t_start, 2)
            return test

        tasks = [get_single_response(test) for test in all_tests]
        all_tests = await asyncio.gather(*tasks)
        elapsed = round(time.time() - t0, 2)
        success = sum(1 for t in all_tests if t.get("actual_output") is not None)
        info(f"[BaseEvaluator] get_test_responses done | success={success}/{len(all_tests)} | elapsed={elapsed}s")
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
        evaluator_name = self.__class__.__name__
        prompt_fqn = state['request'].get('prompt_fqn')
        system_prompt = state['request'].get('system_prompt')
        user_prompt_template = state['request'].get('user_prompt_template')
        t0 = time.time()

        info(f"[{evaluator_name}] verify_tests start | model={self.model_name} | fqn={prompt_fqn or 'pasted'}")

        provided_tests = state['request'].get('test_cases')
        if provided_tests:
            all_tests = provided_tests
            info(f"[{evaluator_name}] Using {len(all_tests)} test cases from request payload")
        elif prompt_fqn:
            all_tests = await self.get_all_tests_db(prompt_fqn)
            all_tests = all_tests[:10]
            info(f"[{evaluator_name}] Fetched {len(all_tests)} test cases from artifact (capped at 10)")
        else:
            error(f"[{evaluator_name}] No test cases provided and no prompt FQN")
            return []

        all_tests = await self.get_test_responses(
            prompt_fqn, all_tests,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )

        evaluation_results = await self.evaluate_tests(all_tests, state['request']['is_rag'])
        info(f"[{evaluator_name}] evaluate_tests done | results={len(evaluation_results)}")

        report = self.get_final_report(all_tests, evaluation_results)
        elapsed = round(time.time() - t0, 2)
        info(f"[{evaluator_name}] verify_tests complete | elapsed={elapsed}s")
        return report