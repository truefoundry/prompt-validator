from langfuse.callback import CallbackHandler

from src.common.config.app_config import get_application_config

CONFIG = get_application_config()


class LangfuseHandler:
    @staticmethod
    def get_handler():
        secret_key = CONFIG.get("telemetry.langfuse.secret_key") or CONFIG.get(
            "LANGFUSE_SECRET_KEY"
        )
        public_key = CONFIG.get("telemetry.langfuse.public_key") or CONFIG.get(
            "LANGFUSE_PUBLIC_KEY"
        )
        host = CONFIG.get("telemetry.langfuse.host") or CONFIG.get("LANGFUSE_HOST")

        if not secret_key or not public_key or not host:
            raise ValueError("Langfuse configuration is incomplete")

        return CallbackHandler(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )
