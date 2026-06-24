import logging
from typing import TypedDict, Annotated

from langgraph.graph.state import CompiledStateGraph

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
from langchain_ollama import ChatOllama

from ..base import Answer
from ...config import Config

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    graphs: list

@tool
def execute_sparql(query: str, state: Annotated[dict, InjectedState]) -> dict[str, list[dict]]:
    """
    Executes a SPARQL query against an endpoint or an rdflib graph.
    :param query: SPARQL query to execute
    :param state: Dictionary containing the graph(s) and the endpoint(s)
    :return: Results of the SPARQL query or an error message
    """
    if "graphs" not in state:
        return {"error": "No graphs provided"}

    total_results = []
    try:
        for g in state["graphs"]:
            results = g.query(query)
            total_results.extend(results)
    except Exception as e:
        return {"error": str(e)}
    return {"results": [dict(row.asdict()) for row in total_results]}

def build(config: Config):
    def assistant(state: MessagesState) -> dict:
        response = llm.invoke([config.var_config.get("system_prompt", "You are a helfpul assistant")]
                              + state["messages"])
        return {"messages": response}

    logging.info(f"Initializing LangGraph backend with config: {config}")

    tools = [
        execute_sparql
    ]

    llm = ChatOllama(
        model=config.var_config.get("model", "gemma4:31b-cloud"),
        temperature=config.var_config.get("temperature", 1.0)
    ).bind_tools(tools)

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.set_entry_point("assistant")
    builder.add_conditional_edges("assistant",
                                  tools_condition,
                                  {"tools": "tools", END: END})
    builder.add_edge("tools", "assistant")

    return builder.compile()

def run_query(llm_graph: CompiledStateGraph, question: str, context: dict) -> Answer:
    logging.info(f"Running query: {question}, context: {context}")
    try:
        # messages = llm_graph.invoke({"messages": question, **context})
        messages = None
        for state in llm_graph.stream(
                {"messages": question, **context},
                stream_mode="values"
        ):
            messages = state
            messages["messages"][-1].pretty_print()
    except Exception as e:
        return Answer(error=str(e))
    return Answer(answer=messages["messages"][-1].content, steps=len(messages["messages"]), trace=messages["messages"])