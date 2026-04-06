from src.chat.utils.redis_async_checkpointer_util import AsyncRedisSaver
from src.common.config.app_config import get_application_config
from src.common.service.logging.logger import info

config = get_application_config()


class CheckpointerFactory:
    @staticmethod
    async def create_checkpointer():
        """Creates and returns an appropriate checkpointer based on configuration."""
        memory_storage_type = config.get("checkpointer.type")
        info(f"Using memory storage type: {memory_storage_type}")

        match memory_storage_type:
            case "redis":
                return await CheckpointerFactory._create_redis_saver()
            case "postgres":
                return await CheckpointerFactory._create_postgres_saver()
            case _:
                info(
                    f"Unsupported storage type: {memory_storage_type}, defaulting to in-memory storage."
                )
                return await CheckpointerFactory._create_redis_saver()

    @staticmethod
    async def _create_redis_saver():
        """Creates a Redis saver instance."""
        try:
            return AsyncRedisSaver.create()
        except Exception as e:
            raise RuntimeError(f"Failed to create RedisSaver: {e}") from e

    @staticmethod
    async def _create_postgres_saver():
        """Creates a Postgres saver instance."""
        try:
            # Lazy import to avoid requiring PostgreSQL libraries when not using Postgres
            from langgraph.checkpoint.postgres import PostgresSaver
            
            return await PostgresSaver.from_conn_string(
                host=config.get("postgres", {}).get("host", "localhost"),
                port=int(config.get("postgres", {}).get("port", 5432)),
                db=config.get("postgres", {}).get("db", "default_db"),
                user=config.get("postgres", {}).get("user", "user"),
                password=config.get("postgres", {}).get("password", "password"),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create PostgresSaver: {e}") from e
