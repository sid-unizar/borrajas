import json
import logging
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState, ToolNode

from borrajas.backends.base import Answer
from borrajas.config import Config


_RDF_SCHEMA = """
Allowed prefixes:
@prefix aemet: <http://aemet.linkeddata.es/ontology/> .
@prefix clasik: <https://example.org/clasik#> .
@prefix geo1: <http://www.w3.org/2003/01/geo/wgs84_pos#> .
@prefix gn: <http://www.geonames.org/ontology#> .
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sorelsc: <http://sweetontology.net/relaSci/> .
@prefix soteria: <https://soteria-public.pages.sintef.no/soteria_ontology_documentation#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

The main node of the knowledge graph is an event, which is an instance of the class clasik:ClimateEvent.
Each event is linked to a country URI through the gn:locatedIn property, which is described by a string through
the property gn:name. Each event is linked to its provinces through the property clasik:isInProvince.
Each event is linked to a start date and an end date through the properties soteria:startDate and soteria:endDate.

Each event is related to an impact through the sorelsc:hasImpact property. Each impact
is characterized as related to deaths through the property clasik:deaths, to people affected through the
property clasik:peopleAffected, to people injured through the property clasik:injured and
to economic loss through the property clasik:economicDamage.

Each event is characterized with a Disaster Type through the property clasik:hasDisasterType,
with a Disaster Subtype through the property clasik:hasDisasterSubtype, and with a Disaster
Subgroup with the property clasik:hasDisasterSubgroup.

If a measurement was recorded, an observation could be linked to an event through the aemet:observedProperty,
and the observation is a quantity value, linked to a string through the property clasik:hasResult.
"""

MAX_EXPLORE_CALLS = 1
MAX_QUERY_CALLS = 4
MAX_RAG_CALLS = 1
MAX_FEEDBACK_ITERATIONS = 2

_RUNTIME_CONTEXTS: dict[str, dict] = {}


def _register_runtime_context(context_id: str, context: dict) -> None:
    _RUNTIME_CONTEXTS[context_id] = {
        "graphs": context.get("graphs", []),
        "rag": context.get("rag"),
    }


def _get_runtime_context(state: dict) -> dict:
    context_id = state.get("context_id")

    if not context_id:
        return {
            "graphs": [],
            "rag": None,
        }

    return _RUNTIME_CONTEXTS.get(
        context_id,
        {
            "graphs": [],
            "rag": None,
        },
    )

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # removed for ChainLit because they are not serializable
    # graphs: list
    # rag: object | None
    # instead, we will use a context register to keep track of runtime contexts
    context_id: str

    order: Literal["kg_first", "rag_first"]
    explore_summary: str
    sparql_result: str
    rag_result: str
    final_answer: str
    feedback_result: Literal["positive", "negative"]
    feedback_count: int

    explore_call_count: int
    query_call_count: int
    rag_call_count: int


def _query_graphs(graphs: list, query: str) -> list[dict]:
    total_results = []

    for graph in graphs:
        results = graph.query(query)
        total_results.extend(dict(row.asdict()) for row in results)

    return total_results


def _serialise_rows(rows: list[dict]) -> list[dict[str, str]]:
    return [
        {
            str(key): str(value)
            for key, value in row.items()
        }
        for row in rows
    ]

def _last_user_question(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage) and not message.content.startswith("EXPLORATION SUMMARY:"):
            return message.content
    return ""

def _last_ai(messages: list[BaseMessage]) -> str:
    return next(
        (
            message.content
            for message in reversed(messages)
            if isinstance(message, AIMessage)
            and message.content
            and not getattr(message, "tool_calls", None)
        ),
        "",
    )


def _invoke_with_retry(llm_bound, messages: list[BaseMessage], retries: int = 3, wait: int = 5):
    for attempt in range(retries):
        try:
            return llm_bound.invoke(messages)
        except Exception:
            if attempt == retries - 1:
                raise

            logging.warning("Transient LLM error. Retry %s/%s in %ss", attempt + 1, retries, wait)
            time.sleep(wait)


def build(config: Config) -> CompiledStateGraph:
    logging.info("Initializing deterministic-feedback LangGraph backend with config: %s", config)

    model = config.params.get("model", "gpt-oss:120b-cloud")
    router_model = config.params.get("router_model", model)
    rag_model = config.params.get("rag_model", model)

    llm = ChatOllama(
        model=model,
        temperature=float(config.params.get("temperature", 1.0)),
        num_ctx=int(config.params.get("num_ctx", 4096)),
    )

    llm_router = ChatOllama(
        model=router_model,
        temperature=float(config.params.get("router_temperature", 0.0)),
        num_ctx=int(config.params.get("num_ctx", 4096)),
    )

    llm_rag_model = ChatOllama(
        model=rag_model,
        temperature=float(config.params.get("rag_temperature", 0.0)),
        num_ctx=int(config.params.get("num_ctx", 4096)),
    )

    @tool
    def run_sparql(query: str, state: Annotated[dict, InjectedState]) -> list[dict]:
        """Execute a SPARQL SELECT query against all configured RDF graphs."""
        try:
            runtime_context = _get_runtime_context(state)
            graphs = runtime_context.get("graphs", [])
            if not graphs:
                return [{"error": "No RDF graphs provided"}]

            return _serialise_rows(_query_graphs(graphs, query))
        except Exception as exc:
            return [{"error": str(exc)}]

    @tool
    def semantic_search(question: str, state: Annotated[dict, InjectedState]) -> str:
        """Semantic search over the configured RAG retriever, if available."""
        runtime_context = _get_runtime_context(state)
        rag = runtime_context.get("rag")

        if rag is None:
            return "RAG is not enabled or no retriever was configured."

        try:
            docs = rag.invoke(question)

            if not docs:
                return "No relevant documents found."

            return "\n\n".join(
                f"SOURCE: {doc.metadata.get('source', '?')}\n"
                f"TITLE: {doc.metadata.get('title', '')}\n"
                f"PASSAGE:\n{doc.page_content}"
                for doc in docs
            )
        except Exception as exc:
            return f"Error: {exc}"

    query_tools = [run_sparql]
    rag_tools = [semantic_search]

    llm_query = llm.bind_tools(query_tools)
    llm_rag = llm_rag_model.bind_tools(rag_tools)

    router_sys = """Route the question to the best knowledge source first.
Reply ONLY with one token:
  kg_first   -> structured event data, counts, dates, countries, impacts, disaster types, deaths, affected people
  rag_first  -> attribution reports, source documents, scientific explanations, climate change role.
Both knowledge sources should be explored when available, independently of the order."""

    kg_query_sys = SystemMessage(
        content=f"""You are a SPARQL writer/executor. Only tool: run_sparql.
You have at most {MAX_QUERY_CALLS} attempts.

Use the KG exploration summary to build a correct SELECT query.
If the first query returns no result, fix URI/predicate/date/string patterns and retry.
No markdown/backticks in query strings.

When you have a useful result, emit a concise natural-language summary.
If all attempts fail or return no evidence, say that the KG evidence is insufficient.

{_RDF_SCHEMA}"""
    )

    rag_sys = SystemMessage(
        content=f"""You are a RAG agent. Only tool: semantic_search.
Call semantic_search at most {MAX_RAG_CALLS} time(s).
Use a specific targeted query.

When summarising, cite the source URL(s) returned by semantic_search.
Separate retrieved report evidence from your own reasoning."""
    )

    synth_sys = SystemMessage(
        content="""You must answer using the evidence provided from the KG/SPARQL component and/or the RAG component.

Return only valid JSON with the following fields:

{
  "short_answer": string,
  "answer": string,
  "kg_evidence": string,
  "rag_evidence": string,
  "sources_used": array of strings,
  "confidence": "high" | "medium" | "low"
}

Rules:
- The short_answer must be the shortest factual answer possible.
- If the question asks for a number, date, country, event ID, disaster type, or yes/no answer, short_answer must contain only that value.
- The answer field may contain one or two explanatory sentences.
- If both KG and RAG contain useful information, combine them.
- If only one source contains useful information, say so.
- Do not add facts not supported by the evidence.
- Reference KG-derived facts in kg_evidence.
- Reference RAG-derived facts in rag_evidence, including source URL if available.
- If the evidence is insufficient, set short_answer to "Insufficient evidence"."""
    )

    feedback_sys = SystemMessage(
        content="""Judge this answer WITHOUT access to source documents.
Score on: relevance, completeness, clarity, citation, consistency.

Reply EXACTLY:
  POSITIVE
or:
  NEGATIVE

Then one sentence explaining your verdict."""
    )

    def router_node(state: AgentState):
        question = _last_user_question(state)

        response = llm_router.invoke(
            [
                SystemMessage(content=router_sys),
                HumanMessage(content=question),
            ]
        )

        order = "kg_first" if "kg_first" in response.content.lower() else "rag_first"
        logging.info("[Router] %s", order)

        return {"order": order}

    def route_after_router(state: AgentState) -> Literal["kg_explore", "rag_agent"]:
        if state["order"] == "kg_first":
            return "kg_explore"

        return "rag_agent"

    def kg_explore_node(state: AgentState):
        runtime_context = _get_runtime_context(state)
        graphs = runtime_context.get("graphs", [])
        question = _last_user_question(state)

        if not graphs:
            return {
                "explore_summary": "No RDF graphs were provided.",
                "messages": [HumanMessage(content="EXPLORATION SUMMARY:\nNo RDF graphs were provided.")],
                "explore_call_count": state.get("explore_call_count", 0) + 1,
            }

        def safe_query(label: str, query: str) -> str:
            try:
                rows = _serialise_rows(_query_graphs(graphs, query))
                return f"{label}:\n{rows[:80]}"
            except Exception as exc:
                return f"{label} error: {exc}"

        classes = safe_query(
            "CLASSES",
            "SELECT DISTINCT ?c WHERE { ?s a ?c } LIMIT 80",
        )

        predicates = safe_query(
            "PREDICATES",
            "SELECT DISTINCT ?p WHERE { ?s ?p ?o } LIMIT 120",
        )

        climate_event_properties = safe_query(
            "PROPERTIES OF clasik:ClimateEvent",
            """
            PREFIX clasik: <https://example.org/clasik#>
            SELECT DISTINCT ?p WHERE {
              ?s a clasik:ClimateEvent ;
                 ?p ?o .
            }
            LIMIT 120
            """,
        )

        sample_events = safe_query(
            "SAMPLE clasik:ClimateEvent INSTANCES",
            """
            PREFIX clasik: <https://example.org/clasik#>
            SELECT DISTINCT ?s WHERE {
              ?s a clasik:ClimateEvent .
            }
            LIMIT 10
            """,
        )

        profile = (
            f"Question: {question}\n\n"
            f"{classes}\n\n"
            f"{predicates}\n\n"
            f"{climate_event_properties}\n\n"
            f"{sample_events}\n\n"
            f"{_RDF_SCHEMA}"
        )

        response = _invoke_with_retry(
            llm,
            [
                SystemMessage(
                    content=(
                        "Produce a concise KG exploration summary for a SPARQL writer. "
                        "Do not call tools. Focus on useful classes, predicates, URI patterns, "
                        "date/impact fields, and constraints relevant to the user question."
                    )
                ),
                HumanMessage(content=profile),
            ],
        )

        summary = response.content or "No KG exploration summary was produced."

        logging.info("[Explore] completed")

        return {
            "explore_summary": summary,
            "messages": [HumanMessage(content="EXPLORATION SUMMARY:\n" + summary)],
            "explore_call_count": state.get("explore_call_count", 0) + 1,
        }

    def kg_query_node(state: AgentState):
        messages = state["messages"]
        question = HumanMessage(content=_last_user_question(state))

        exploration = HumanMessage(
            content="EXPLORATION SUMMARY:\n" + state.get("explore_summary", "")
        )

        relevant_history = [
            message
            for message in messages
            if (
                isinstance(message, ToolMessage)
                and getattr(message, "name", "") == "run_sparql"
            )
            or (
                isinstance(message, AIMessage)
                and any(
                    tool_call["name"] == "run_sparql"
                    for tool_call in getattr(message, "tool_calls", [])
                )
            )
        ][-6:]

        response = _invoke_with_retry(
            llm_query,
            [kg_query_sys, question, exploration] + relevant_history,
        )

        return {
            "messages": [response],
            "query_call_count": state.get("query_call_count", 0) + 1,
        }

    def kg_query_condition(state: AgentState) -> Literal["query_tools", "store_kg"]:
        last = state["messages"][-1]
        calls = state.get("query_call_count", 0)

        if getattr(last, "tool_calls", None) and calls < MAX_QUERY_CALLS:
            return "query_tools"

        return "store_kg"

    def store_kg_result(state: AgentState):
        result = _last_ai(state["messages"]) or "No KG answer."

        logging.info("[KG Query] %s", result[:200])

        return {
            "sparql_result": result,
        }

    def route_after_kg(state: AgentState) -> Literal["rag_agent", "synthesizer"]:
        runtime_context = _get_runtime_context(state)
        rag = runtime_context.get("rag")

        if state["order"] == "kg_first" and rag is not None:
            return "rag_agent"

        return "synthesizer"

    def rag_agent_node(state: AgentState):
        question = _last_user_question(state)

        retrieval_query = question

        if state.get("sparql_result"):
            retrieval_query = (
                f"{question}\n\n"
                f"KG result to contextualize retrieval:\n{state['sparql_result']}"
            )
        elif state.get("explore_summary"):
            retrieval_query = (
                f"{question}\n\n"
                f"KG exploration summary to contextualize retrieval:\n{state['explore_summary']}"
            )

        response = _invoke_with_retry(
            llm_rag,
            [
                rag_sys,
                HumanMessage(content=retrieval_query),
            ],
        )

        return {
            "messages": [response],
            "rag_call_count": state.get("rag_call_count", 0) + 1,
        }

    def rag_tools_condition(state: AgentState) -> Literal["rag_tools", "store_rag"]:
        last = state["messages"][-1]
        calls = state.get("rag_call_count", 0)

        if getattr(last, "tool_calls", None) and calls < MAX_RAG_CALLS:
            return "rag_tools"

        return "store_rag"

    def store_rag_result(state: AgentState):
        result = _last_ai(state["messages"]) or "No RAG answer."

        logging.info("[RAG] %s", result[:200])

        return {
            "rag_result": result,
        }

    def route_after_rag(state: AgentState) -> Literal["kg_explore", "synthesizer"]:
        if state["order"] == "rag_first":
            return "kg_explore"

        return "synthesizer"

    def synthesizer_node(state: AgentState):
        question = _last_user_question(state)

        prompt = (
            f"Question: {question}\n\n"
            f"## KG result\n{state.get('sparql_result', 'N/A')}\n\n"
            f"## RAG result\n{state.get('rag_result', 'N/A')}"
        )

        response = _invoke_with_retry(
            llm,
            [
                synth_sys,
                HumanMessage(content=prompt),
            ],
        )

        logging.info("[Synthesizer] attempt %s", state.get("feedback_count", 0) + 1)

        return {
            "messages": [response],
            "final_answer": response.content,
        }

    def feedback_node(state: AgentState):
        question = _last_user_question(state)

        response = _invoke_with_retry(
            llm,
            [
                feedback_sys,
                HumanMessage(
                    content=(
                        f"Question: {question}\n\n"
                        f"Answer:\n{state.get('final_answer', '')}"
                    )
                ),
            ],
        )

        lines = response.content.strip().splitlines()
        verdict = "positive" if lines and "POSITIVE" in lines[0].upper() else "negative"

        logging.info("[Feedback] %s", verdict)

        return {
            "feedback_result": verdict,
            "feedback_count": state.get("feedback_count", 0) + 1,
        }

    def reset_for_retry(state: AgentState):
        logging.info("[Reset] retry #%s", state.get("feedback_count", 0))

        return {
            "explore_summary": "",
            "sparql_result": "",
            "rag_result": "",
            "final_answer": "",
            "explore_call_count": 0,
            "query_call_count": 0,
            "rag_call_count": 0,
        }

    def route_after_feedback(state: AgentState) -> Literal["reset_for_retry", "__end__"]:
        if state.get("feedback_result") == "positive":
            return "__end__"

        if state.get("feedback_count", 0) >= MAX_FEEDBACK_ITERATIONS:
            return "__end__"

        return "reset_for_retry"

    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("kg_explore", kg_explore_node)
    builder.add_node("kg_query", kg_query_node)
    builder.add_node("query_tools", ToolNode(query_tools))
    builder.add_node("store_kg", store_kg_result)
    builder.add_node("rag_agent", rag_agent_node)
    builder.add_node("rag_tools", ToolNode(rag_tools))
    builder.add_node("store_rag", store_rag_result)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("feedback", feedback_node)
    builder.add_node("reset_for_retry", reset_for_retry)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "kg_explore": "kg_explore",
            "rag_agent": "rag_agent",
        },
    )

    builder.add_edge("kg_explore", "kg_query")

    builder.add_conditional_edges(
        "kg_query",
        kg_query_condition,
        {
            "query_tools": "query_tools",
            "store_kg": "store_kg",
        },
    )

    builder.add_edge("query_tools", "kg_query")

    builder.add_conditional_edges(
        "store_kg",
        route_after_kg,
        {
            "rag_agent": "rag_agent",
            "synthesizer": "synthesizer",
        },
    )

    builder.add_conditional_edges(
        "rag_agent",
        rag_tools_condition,
        {
            "rag_tools": "rag_tools",
            "store_rag": "store_rag",
        },
    )

    builder.add_edge("rag_tools", "rag_agent")

    builder.add_conditional_edges(
        "store_rag",
        route_after_rag,
        {
            "kg_explore": "kg_explore",
            "synthesizer": "synthesizer",
        },
    )

    builder.add_edge("synthesizer", "feedback")

    builder.add_conditional_edges(
        "feedback",
        route_after_feedback,
        {
            "reset_for_retry": "reset_for_retry",
            "__end__": END,
        },
    )

    builder.add_edge("reset_for_retry", "router")

    return builder.compile(checkpointer=MemorySaver())


def _format_answer(content: str) -> str:
    if not content:
        return "No answer returned."

    try:
        parsed = json.loads(content)

        answer = parsed.get("answer") or parsed.get("short_answer") or content
        short_answer = parsed.get("short_answer")
        confidence = parsed.get("confidence")
        sources = parsed.get("sources_used") or []

        parts = []

        if short_answer:
            parts.append(f"**Short answer:** {short_answer}")

        parts.append(str(answer))

        if confidence:
            parts.append(f"**Confidence:** {confidence}")

        if sources:
            parts.append(
                "**Sources used:**\n"
                + "\n".join(f"- {source}" for source in sources)
            )

        return "\n\n".join(parts)
    except Exception:
        return content


def run_query(llm_graph: CompiledStateGraph, question: str, context: dict) -> Answer:
    logging.info("Running deterministic-feedback query: %s", question)

    thread_id = context.get("thread_id", "default")
    context_id = thread_id

    _register_runtime_context(context_id, context)

    invoke_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    input_state = {
        "messages": [HumanMessage(content=question)],
        "context_id": context_id,
        "order": "kg_first",
        "explore_summary": "",
        "sparql_result": "",
        "rag_result": "",
        "final_answer": "",
        "feedback_result": "negative",
        "feedback_count": 0,
        "explore_call_count": 0,
        "query_call_count": 0,
        "rag_call_count": 0,
    }

    try:
        result = llm_graph.invoke(input_state, invoke_config)

        final_answer = result.get("final_answer") or _last_ai(result.get("messages", []))
        formatted_answer = _format_answer(final_answer)

        return Answer(
            answer=formatted_answer,
            steps=len(result.get("messages", [])),
            trace=result.get("messages", []),
            config={
                "thread_id": thread_id,
                "context_id": context_id,
                "feedback_count": result.get("feedback_count", 0),
                "feedback_result": result.get("feedback_result"),
                "order": result.get("order"),
            },
        )
    except Exception as exc:
        logging.exception("Error while running deterministic-feedback query")
        return Answer(error=str(exc))