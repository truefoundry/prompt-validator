from deepeval.metrics import ContextualPrecisionMetric, ToxicityMetric, BiasMetric, GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval import evaluate
from deepeval.evaluate import ErrorConfig
import os
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI

from src.chat.graph.base_evaluator import PromptEvaluator
from src.chat.utils.llm_models import get_truefoundry_llm
from langchain_openai import AzureChatOpenAI
from src.common.service.logging.logger import info

class DeepEvalPromptEvaluator(PromptEvaluator):
    """Evaluator that uses DeepEval metrics for evaluation."""
    
    async def get_all_metrics(self, is_rag):
        """Get all DeepEval metrics."""
        # get the llm to be used as judge
        # custom_llm = AzureChatOpenAI(get_truefoundry_llm())

        # Define the different metrics
        correctness_metric = GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=self.config.get("metrics.prompt_metrics.correctness_geval.threshold"),
            async_mode=False
        )
        
        toxicity_metric = ToxicityMetric(
            threshold=self.config.get("metrics.prompt_metrics.toxicity_metric.threshold"), 
            async_mode=False
        ) 
        
        bias_metric = BiasMetric(
            threshold=self.config.get("metrics.prompt_metrics.bias_metric.threshold"), 
            async_mode=False
        )

        rag_metrics = []
        """
        if is_rag:
            context_precision = ContextualPrecisionMetric(
                model=custom_llm, 
                threshold=self.config.get("metrics.rag_metrics.context_precision.threshold"), 
                async_mode=False
            )
            rag_metrics = [context_precision]
        """

        return [(correctness_metric, correctness_metric.__name__),
                (toxicity_metric, "Toxicity"),
                (bias_metric, "Bias")] + rag_metrics
    
    async def evaluate_tests(self, all_tests, is_rag=False):
        """Evaluate tests using DeepEval metrics."""
        # Get all metrics
        all_metrics = await self.get_all_metrics(is_rag)
        info(f"All metrics: {all_metrics}")

        llm_test_cases = list()
        for test_case in all_tests:
            """
            if is_rag:
                llm_test_cases.append(
                    LLMTestCase(
                        input=test_case["input"],
                        actual_output=test_case["actual_output"],
                        expected_output=test_case["expected_output"],
                        retrieval_context=test_case["retrieval_context"]
                        )
                    )
            else:
            """
            llm_test_cases.append(
                LLMTestCase(
                    input=test_case["data"]["input"],
                    actual_output=test_case["actual_output"],
                    expected_output=test_case["expected_output"]
                    )
                )

        info("LLM test cases created.")

        error_config = ErrorConfig(ignore_errors=True)
        result = evaluate(llm_test_cases, [m[0] for m in all_metrics], error_config=error_config)
        info("Evaluation completed.")
        test_results = result.test_results
        test_results.sort(key=lambda x: int(x.name.split("_")[-1]))

        evaluation_results = list()
        for test_case, result in zip(all_tests, test_results):
            metric_values = dict()
            pass_status = True
            for name, metric in zip([m[1] for m in all_metrics], result.metrics_data):
                pass_status = pass_status and metric.success
                metric_values[name] = {
                    "score": metric.score,
                    "reason": metric.reason,
                    "threshold": metric.threshold,
                    "success": metric.success
                }
            evaluation_results.append({
                            "test_case_id": test_case.get('test_case_id'),
                            "input": test_case.get("input"),
                            "actual_output": test_case.get("actual_output"),
                            "expected_output": test_case.get("expected_output"),
                            "scenario": test_case.get("scenario"),
                            "metric_values": metric_values,
                            "pass_status": pass_status
                        })
        return evaluation_results
    
    def get_final_report(self, all_tests, evaluation_results):
        """Generate report with DeepEval metrics."""
        metric_tests = [test for test in evaluation_results if ('metric_values' in test.keys())]
        metric_values = {
            "Number of Tests": len(all_tests),
            "Number of Tests with Errors": len(all_tests) - len(metric_tests)
        }

        count = len(all_tests)
        for metric_name in metric_tests[0]['metric_values'].keys() if metric_tests else []:
            metric_values[f"Average {metric_name}"] = sum(
                [test['metric_values'][metric_name]['score'] for test in metric_tests]
            ) / count
        
        return {"all_tests": evaluation_results, "all_metrics": metric_values} 