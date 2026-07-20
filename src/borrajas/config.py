import logging
import os
import tomllib

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

@dataclass
class Config:
    backend: str = None
    experiment: str = None
    endpoints: list[str] = field(default_factory=list)
    ttl: list[str] = field(default_factory=list)
    log_level: str = "INFO"

    # Additional parameters specific to the experiment
    params: dict = field(default_factory=dict)

    @property
    def full_exp_name(self) -> str:
        return f"{self.backend}/{self.experiment}"

    @property
    def kg_enabled(self) -> bool:
        return len(self.endpoints) > 0 or len(self.ttl) > 0

    @property
    def rag_enabled(self) -> bool:
        return self.params.get("rag_enabled", False)

    @staticmethod
    def _load_env_config():
        def normalize(var: str) -> str:
            return var.replace("BORRAJAS_EXP_", "").lower()

        load_dotenv()

        endpoints_env = os.getenv("BORRAJAS_ENDPOINTS")
        ttl_env = os.getenv("BORRAJAS_TTL_FILES")

        values = {
            "backend": os.getenv("BORRAJAS_BACKEND"),
            "experiment": os.getenv("BORRAJAS_EXPERIMENT"),
            "endpoints": endpoints_env.split(",") if endpoints_env else None,
            "ttl": ttl_env.split(",") if ttl_env else None,
            "log_level": os.getenv("BORRAJAS_LOG_LEVEL"),
            "params": {
                normalize(var): value for var, value in os.environ.items() if var.startswith("BORRAJAS_EXP_")
            }
        }

        return {k: v for k, v in values.items() if v is not None}

    @classmethod
    def _filter_valid_fields(cls, cli_args: dict[str, Any]):
        valid_fields = {f.name for f in fields(cls)}
        return {k: v for k, v in cli_args.items() if v is not None and k in valid_fields}

    @staticmethod
    def _normalise(config: dict):
        config = config.copy()  # avoiding side effects

        # parsing input like backend=langgraph/react
        if not config.get("backend") and config.get("variant") and config["backend"].count("/") == 1:
            logging.info(f"Parsing variant name. Backend: {config["backend"]}, variant: {config['variant']}")
            config["backend"], config["variant"] = config["variant"].split("/", 1)

        return config

    @staticmethod
    def _validate(config: dict):
        if not config.get("backend"):
            raise ValueError("Backend not specified")
        if not config.get("experiment"):
            raise ValueError("Experiment not specified")
        if config["log_level"] not in logging.getLevelNamesMapping():
            raise ValueError(f"Invalid log level: {config['log_level']}")
        if not len(config.get("ttl", [])) and not len(config.get("endpoints", [])):
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

        config = cls._normalise(config)
        cls._validate(config)

        return cls(**config)
