import uuid
from typing import Dict

from src.common.util.context_util import get_request_execution_ctx


def get_grid() -> str:
    return get_request_execution_ctx().grid or str(uuid.uuid4())


def default_headers() -> dict:
    return {"Content-Type": "application/json", "x-grid": get_grid()}


def get_downstream_request_headers(received_headers) -> Dict[str, str]:
    # if there is any header starting with 'x-', include them in the request
    x_headers = {
        k: v for k, v in received_headers.items() if k.lower().startswith("x-")
    }

    return {**default_headers(), **x_headers}
