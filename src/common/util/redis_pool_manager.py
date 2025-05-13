import threading

from redis.asyncio import Redis, ConnectionPool, SSLConnection

from src.common.config.app_config import get_application_config
from src.common.service.logging.logger import signal

config = get_application_config()


class RedisPoolManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(RedisPoolManager, cls).__new__(
                        cls, *args, **kwargs
                    )
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            signal(f"Initializing RedisPoolManager")
            self.redis_pool = _create_connection_pool()
            self.initialized = True

    async def get_redis_connection(self) -> Redis:
        """
        Get a Redis connection from the pool.
        Args:
            None
        Returns:
            Redis connection
        """
        return Redis(connection_pool=self.redis_pool)

    async def close_redis_pool(self):
        """
        Close the Redis connection pool.
        Args:
            None
        Returns:
            None
        """
        if self.redis_pool:
            try:
                await self.redis_pool.disconnect()
            except Exception as e:
                raise Exception(f"Failed to disconnect Redis connection pool: {e}")
            finally:
                self.redis_pool = None


def _create_connection_pool() -> ConnectionPool:
    """
    Create a connection pool for Redis.
    """
    redis_password = config.get("CHAT_HISTORY_REDIS_STORE_PASSWORD", "")
    redis_cert = config.get("redis.cert_path", "")
    redis_host = config.get("redis.host", "localhost")
    redis_port = config.get("redis.port", 6379)
    redis_db = config.get("redis.db", 0)
    max_connections = config.get("redis.max_connections", None)
    socket_timeout = config.get("redis.socket_timeout", None)
    socket_connect_timeout = config.get("redis.socket_connect_timeout", None)
    local_hosts = {None, "localhost", "0.0.0.0", "127.0.0.1"}

    if not redis_host:
        raise ValueError("Redis host must be specified.")

    if not redis_port:
        raise ValueError("Redis port must be specified.")

    if not isinstance(redis_port, int):
        raise ValueError("Redis port must be an integer.")

    if not isinstance(redis_db, int):
        raise ValueError("Redis database index must be an integer.")

    if (
        redis_password
        and len(redis_password) > 3
        and redis_cert
        # and redis_host not in local_hosts
    ):
        signal("Redis SSL enabled")
        conn_pool = ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            connection_class=SSLConnection,
            ssl_cert_reqs="required",
            ssl_ca_certs=redis_cert,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
        )
    else:
        signal("Redis SSL disabled")
        conn_pool = ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password if redis_password else None,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
        )

    return conn_pool
