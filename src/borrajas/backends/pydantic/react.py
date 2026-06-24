import logging
from ..base import Answer

def build(config: dict):
    logging.info(f"Initializing Pydantic-AI backend with config: {config}")

def run_query(question: str, config: dict, environment: dict) -> Answer:
    return Answer(error="Not implemented")