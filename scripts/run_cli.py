import argparse

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

    parser.add_argument("-b", "--backend", required=True,
                        choices=["langgraph", "pydantic"],
                        help="The backend to use")

    parser.add_argument("--endpoint", required=False,
                        help="The SPARQL endpoint to use as a knowledge graph")
    parser.add_argument("--ttl", required=False, action="append",
                        help="The Turtle file to use as a knowledge graph, this parameter can be used multiple times")

    parser.add_argument("-c", "--config", required=False,
                        help="Path to the config file")

    parser.add_argument("--version", action="version", version="%(prog)s 0.1",
                        help="Show version information and exit")

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.question:
        print("There is only one question: ", args.question)
    if args.questions:
        print("There are multiple questions: ", args.questions)

if __name__ == "__main__":
    main()