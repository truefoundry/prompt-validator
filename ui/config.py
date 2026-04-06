import os


DEFAULT_BASE_URL = os.getenv("PROMPT_TUNER_BASE_URL", "http://localhost:21120")
DEFAULT_GRID = "8091"
DEFAULT_MODEL_NAME = os.getenv("PROMPT_TUNER_MODEL_NAME", "tfy-ai-vertex/gemini-3-flash-preview")
REQUEST_TIMEOUT_SECONDS = 600
