import logging
import os
from functools import lru_cache
from functools import reduce

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

with open(
    os.path.join(os.environ["CONFIGBASEPATH"], "application.yaml"), "r"
) as stream:
    try:
        yaml_settings: dict = dict(yaml.safe_load(stream))
        env_settings = dict(os.environ)
        yaml_settings.update(env_settings)

    except OSError as e:
        print(f"Unable to open application yaml file: {e}")
    except Exception as ex:
        print(f"Unknown Exception in loading application yaml file : {ex}")


class Settings(BaseSettings):
    """Application Config Settings"""

    APP: dict = yaml_settings

    def get(self, keys: str, default=None):
        return deep_get(yaml_settings, keys, default)


settings = Settings()


@lru_cache()
def get_application_config():
    """Get Application Config"""
    return settings


def deep_get(dictionary, keys, default=None):
    """Deep Get Function to Flatten nested dictionary structure"""
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys.split("."),
        dictionary,
    )


def filter_maker(level):
    """Filter Maker For Application Logging"""
    level = getattr(logging, level)

    def filter(record):
        return record.levelno <= level

    return filter
