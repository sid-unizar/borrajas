from dataclasses import dataclass, field

@dataclass
class Answer:
    answer: str = None
    error: str = None
    steps: int = 0
    trace: list = field(default_factory=list)
    config: dict = field(default_factory=dict)

# def init(config: dict):
#     print(f"Initializing LangGraph backend with config: {config}")
#
# def run_query(question: str, config: dict, environment: dict) -> Answer:
#     return Answer(error="Not implemented")