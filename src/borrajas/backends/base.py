from dataclasses import dataclass

@dataclass
class Answer:
    answer: str = None
    error: str = None

def run_query(question: str, config: dict, environment: dict) -> Answer:
    return Answer(error="Not implemented")