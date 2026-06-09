import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
import tomllib

from borrajas.backends import BACKENDS

@dataclass
class Config:
    backend: str = "langgraph"
    endpoint: list[str] = field(default_factory=list)
    ttl: list[str] = field(default_factory=list)
    log_level: str = "INFO"

    @staticmethod
    def _load_env_config():
        values = {
            "backend": os.getenv("BORRAJAS_BACKEND"),
            "endpoint": os.getenv("BORRAJAS_ENDPOINT").split(",") if os.getenv("BORRAJAS_ENDPOINT") else None,
            "ttl": os.getenv("BORRAJAS_TTL_FILES").split(",") if os.getenv("BORRAJAS_TTL_FILES") else None,
            "log_level": os.getenv("BORRAJAS_LOG_LEVEL"),
        }

        return {k: v for k, v in values.items() if v is not None}

    @classmethod
    def _filter_valid_fields(cls, cli_args: dict[str, Any]):
        valid_fields = {f.name for f in fields(cls)}
        return {k: v for k, v in cli_args.items() if v is not None and k in valid_fields}

    @staticmethod
    def _validate(config: dict):
        if config["backend"] not in BACKENDS:
            raise ValueError(f"Invalid backend: {config['backend']}")
        if config["log_level"] not in logging.getLevelNamesMapping():
            raise ValueError(f"Invalid log level: {config['log_level']}")
        if config["endpoint"] and not config["ttl"]:
            logging.warning("No endpoints nor turtle files provided, possibly a configuration error.")

    @classmethod
    def load_config(cls,
                    filename: str | Path | None = None,
                    cli_args: dict[str, Any] | None = None):
        conf_obj = cls()
        config = conf_obj.__dict__

        # First pass: config file
        if filename:
            with open(filename, "rb") as f:
                config.update(cls._filter_valid_fields(tomllib.load(f)))

        # Overriding config file with environment variables
        config.update(cls._load_env_config())

        # Overriding config file and environment variables with command line arguments
        if cli_args:
            config.update(cls._filter_valid_fields(cli_args))

        cls._validate(config)

        return cls(**config)
