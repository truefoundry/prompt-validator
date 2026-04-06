import os

BASE_URL = os.getenv("EVAL_BASE_URL", "http://localhost:21120")
MODEL_NAME = os.getenv("EVAL_MODEL_NAME", None)  # None → server default
CONCURRENCY_LIMIT = int(os.getenv("EVAL_CONCURRENCY", "5"))
REQUEST_TIMEOUT_S = float(os.getenv("EVAL_TIMEOUT_S", "120"))
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set", "prompts.json")
DEFAULT_JUDGE_METRICS = ["clarity", "completeness", "accuracy", "conciseness", "professional_tone"]
