import argparse
import logging
import sys

from rdflib import Graph
from rdflib.plugins.stores.sparqlstore import SPARQLStore

from .backends import VARIANTS
from .config import Config

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="borrajas",
        description="Command line interface for the RDF+RAG-enabled hybrid chatbot for extreme climate events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Calling borrajas with the LangGraph backend, SINOBAS dataset, and a single question
    borrajas-cli --backend langgraph --ttl data/sinobas-sample.ttl "How many tornadoes have occurred in the last 10 years in Cantabria?"

    # Calling borrajas with the LangGraph backend, a SPARQL endpoint, and a list of questions
    borrajas-cli --backend langgraph --endpoint https://sparql.ionov.me/clasik -i data/sample-questions.txt

    # Calling borrajas with a config file and a list of questions
    borrajas-cli --config config.yaml --questions data/sample-questions.txt 
    """
    )

    q_group = parser.add_mutually_exclusive_group(required=True)
    q_group.add_argument("-i", "--questions", required=False,
                         help="A filename with the list of questions to ask")
    q_group.add_argument("-q", "--question", required=False,
                         help="A question to ask")

    parser.add_argument("-b", "--backend", required=False,
                        choices=VARIANTS.keys(),
                        help="The backend to use")

    parser.add_argument("-v", "--variant", required=False,
                        help="The variant of the prototype to use (e.g. react)")

    parser.add_argument("--endpoint", required=False, action="append",
                        help="The SPARQL endpoints to use as knowledge graphs, this parameter can be used multiple times")
    parser.add_argument("--ttl", required=False, action="append",
                        help="The Turtle file to use as a knowledge graph, this parameter can be used multiple times")

    parser.add_argument("-c", "--config", required=False,
                        help="Path to the config file")

    parser.add_argument("-l", "--log-level", required=False,
                        dest="log_level",
                        choices=logging.getLevelNamesMapping().keys(),
                        help="The log level to use")

    parser.add_argument("--version", action="version", version="%(prog)s 0.1",
                        help="Show version information and exit")

    return parser

def load_questions(filename: str) -> list[str]:
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]

def main():
    parser = create_parser()
    args = parser.parse_args()

    try:
        config = Config.load_config(args.config, vars(args))
    except ValueError as e:
        print(f"Invalid configuration: {e}", file=sys.stderr)
        return
    # If we are still here, we can assume that the config is valid
    logging.basicConfig(level=config.log_level, format="%(levelname)s: %(message)s")
    logging.info(f"Loaded configuration: {config}")

    full_name = f"{config.backend}/{config.variant}"
    agent = VARIANTS[full_name].build(config)
    context = {
        "graphs": [Graph().parse(ttl, format="turtle") for ttl in config.ttl] +
                  [Graph(store=SPARQLStore(endpoint)) for endpoint in config.endpoint],
    }

    logging.info(f"Context: {context}")

    questions = load_questions(args.questions) if args.questions else [args.question]

    for question in questions:
        logging.info(f"Question: {question}")
        answer = VARIANTS[full_name].run_query(agent, question, context)
        if answer.error:
            logging.error(f"Error: {answer.error}")
            continue
        logging.debug(f"Trace: {answer.trace}")

        print(answer.answer)

if __name__ == "__main__":
    main()
