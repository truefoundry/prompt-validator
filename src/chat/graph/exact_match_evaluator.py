import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.chat.graph.base_evaluator import PromptEvaluator
from src.common.service.logging.logger import info

class ExactMatchEvaluator(PromptEvaluator):
    """Evaluator that performs exact matching between expected and actual outputs."""
    
    async def evaluate_tests(self, all_tests, is_rag=False):
        """Evaluate tests by exact matching."""
        for test in all_tests:
            test["pass_status"] = (test['actual_output'] == test['expected_output'])
        return all_tests
    
    def get_final_report(self, all_tests, evaluation_results=None):
        """Generate report with exact matching metrics."""
        y_true = [test['actual_output'] for test in all_tests if (test['actual_output'] is not None)]
        y_pred = [test['expected_output'] for test in all_tests if (test['actual_output'] is not None)]

        # Get the labels (class names)
        labels = list(set(y_true))

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )

        cf = confusion_matrix(y_true, y_pred, labels=labels)

        # Print scores per class
        metrics = dict()
        for i, label in enumerate(labels):
            metrics[label] = {
                "Number of Tests": int(support[i]),
                "Accuracy": float(cf[i][i]/support[i]),
                "Precision": float(precision[i]),
                "Recall": float(recall[i]),
                "F1 Score": float(f1[i])}
        
        all_metrics = {
            "Number of Tests": len(y_true),
            "Number of Tests with Errors": len([test_case for test_case in all_tests if (test_case['actual_output'] is None)]),
            "Accuracy": float(cf.diagonal().sum()/len(y_true)),
            "Average Precision": float(np.average(precision)),
            "Average Recall": float(np.average(recall)),
            "Average F1 Score": float(np.average(f1)),
            "Classwise Metrics": metrics}
        
        info(f"all_metrics: {all_metrics}")
        return {"all_tests": all_tests, "all_metrics": all_metrics} 