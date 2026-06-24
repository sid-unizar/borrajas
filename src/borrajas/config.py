import logging
import os
import tomllib

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

@dataclass
class Config:
    backend: str = "langgraph"
    variant: str = None
    endpoint: list[str] = field(default_factory=list)
    ttl: list[str] = field(default_factory=list)
    log_level: str = "INFO"
    var_config: dict = field(default_factory=dict)

    @staticmethod
    def _load_env_config(predefined: bool = True):
        def normalize(var: str) -> str:
            return var.replace("BORRAJAS_", "").lower()

        load_dotenv()

        values = {
            "backend": os.getenv("BORRAJAS_BACKEND"),
            "variant": os.getenv("BORRAJAS_VARIANT"),
            "endpoint": os.getenv("BORRAJAS_ENDPOINT").split(",") if os.getenv("BORRAJAS_ENDPOINT") else None,
            "ttl": os.getenv("BORRAJAS_TTL_FILES").split(",") if os.getenv("BORRAJAS_TTL_FILES") else None,
            "log_level": os.getenv("BORRAJAS_LOG_LEVEL"),
        } if predefined else {normalize(var): value for var, value in os.environ.items() if var.startswith("BORRAJAS_")}

        return {k: v for k, v in values.items() if v is not None}

    @classmethod
    def _filter_valid_fields(cls, cli_args: dict[str, Any]):
        valid_fields = {f.name for f in fields(cls)}
        return {k: v for k, v in cli_args.items() if v is not None and k in valid_fields}

    @staticmethod
    def _clean_and_validate(config: dict):
        if not config.get("backend") and config.get("variant") and config["variant"].count("/") == 1:
            logging.info(f"Parsing variant name. Backend: {config["backend"]}, variant: {config['variant']}")
            config["backend"], config["variant"] = config["variant"].split("/")[0]
        if not config.get("backend"):
            raise ValueError("Backend not specified")
        if not config.get("variant"):
            raise ValueError("Variant not specified")
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
        config["var_config"].update( cls._load_env_config(predefined=False))

        # Overriding config file and environment variables with command line arguments
        if cli_args:
            config.update(cls._filter_valid_fields(cli_args))

        cls._clean_and_validate(config)

        return cls(**config)
