import requests
from abc import ABC, abstractmethod

from src.common.config.app_config import get_application_config
from src.common.service.llm_prompt.prompt_service import PromptService
from src.common.service.logging.logger import error, info

class PromptEvaluator(ABC):
    """Base class for prompt evaluation strategies."""
    
    def __init__(self):
        self.config = get_application_config()
    
    async def get_all_tests_db(self, prompt_fqn):
        """Fetch test cases from the database."""
        url = f"{self.config.get('client.test_cases_db.url')}/{prompt_fqn}"
        headers = {
            'accept': 'application/json'
        }

        try:
            response = requests.get(url, headers=headers)
            all_tests = response.json()
            info(f"Number of test cases from DB: {len(all_tests)}")
            return all_tests
        except requests.exceptions.RequestException as e:
            error(f"Error fetching test cases from API: {str(e)}")
            return []
    
    async def get_test_responses(self, prompt_fqn, all_tests):
        """Get responses for all test cases."""
        for test in all_tests:
            response = await PromptService.get_prompt_response(prompt_fqn=prompt_fqn, data=test["data"])
            test["actual_output"] = response["aiResponse"]["content"]
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
        # Get prompt details
        prompt_fqn = state['request']['prompt_fqn']
        prompt_detail = await PromptService.get_prompt_details([prompt_fqn])

        # Get all test cases
        all_tests = await self.get_all_tests_db(prompt_fqn)
        info("test cases created")
        
        # Get all actual outputs
        all_tests = await self.get_test_responses(prompt_fqn, all_tests)
        
        # Evaluate tests using the specific strategy
        evaluation_results = await self.evaluate_tests(all_tests, prompt_detail[0].rag)
        
        # Generate final report
        report = self.get_final_report(all_tests, evaluation_results)
        
        return report 