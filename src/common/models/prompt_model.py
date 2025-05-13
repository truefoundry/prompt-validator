import base64
from typing import List, Optional, Dict, Any

from pydantic import BaseModel

from src.common.service.logging.logger import error


class PromptDetail(BaseModel):
    def __init__(self, **data: Any):
        super().__init__(**data)
        self.decode_prompt()

    def decode_prompt(self):
        """Decode the base64 encoded prompt and reset its value."""
        try:
            decoded_bytes = base64.b64decode(self.generatedPrompt)
            self.generatedPrompt = decoded_bytes.decode("utf-8")

        except (base64.binascii.Error, UnicodeDecodeError) as e:
            error(f"Error decoding prompt: {e}")

    promptId: str
    model: str
    provider: str
    prompt: str
    generatedPrompt: str
    parameters: Dict[str, Any]
    context: Optional[str] = None
    intent: Optional[str] = None
    functions: Optional[str] = None
    agentName: Optional[str] = None
    dataStore: Optional[str] = None
    rag: Optional[bool] = None
    agent: Optional[bool] = None
    feedback: Optional[bool] = None
    user: Optional[str] = None
    createTs: Optional[str] = None
    updateTs: Optional[str] = None


class PromptDetails(BaseModel):
    statusCode: str
    statusDescription: str
    fault: Optional[Any] = None
    data: List[PromptDetail]
