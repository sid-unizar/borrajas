import logging
import uuid
from dataclasses import dataclass

import chainlit as cl
from rdflib import Graph
from rdflib.plugins.stores.sparqlstore import SPARQLStore

from borrajas.backends.langgraph import deterministic_feedback
from borrajas.config import Config
from borrajas.rag import init_rag

BACKEND = "langgraph"
EXPERIMENT = "default"
ENDPOINTS: list[str] = []
TTL_FILES = ["data/emdat_climate_events.ttl"]
LOG_LEVEL = "INFO"
RAG_URLS_PATH = "data/wwa_urls.txt"

@dataclass(frozen=True)
class AppState:
    config: Config
    agent: object
    context: dict

def build_config() -> Config:
    return Config(
        backend=BACKEND,
        experiment=EXPERIMENT,
        ttl=TTL_FILES,
        endpoints=ENDPOINTS,
        log_level=LOG_LEVEL,
        params = {
            "rag_enabled": RAG_URLS_PATH is not None,
            "rag_urls_path": RAG_URLS_PATH
        }
    )

def build_context(config: Config) -> dict:
    return {
        "graphs": [Graph().parse(ttl, format="turtle") for ttl in config.ttl] +
                  [Graph(store=SPARQLStore(endpoint)) for endpoint in config.endpoints],
        "rag": init_rag(config) if config.rag_enabled else None
    }

def initialise() -> AppState:
    config = build_config()

    logging.basicConfig(
        level=config.log_level,
        format="%(levelname)s: %(message)s",
    )

    logging.info("Initialising app")
    logging.info("Config: %s", config)

    agent = deterministic_feedback.build(config)
    context = build_context(config)

    return AppState(
        config=config,
        agent=agent,
        context=context
    )

APP_STATE = initialise()

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", str(uuid.uuid4()))

    await cl.Message(
        content="Ready to answer questions on extreme weather events"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    if not thread_id:
        thread_id = str(uuid.uuid4())
        cl.user_session.set("thread_id", thread_id)
    status = cl.Message(content="Thinking...")
    await status.send()

    context = {
        **APP_STATE.context,
        "thread_id": thread_id
    }

    try:
        ans = await cl.make_async(deterministic_feedback.run_query)(
            APP_STATE.agent,
            message.content,
            context
        )

        status.content = ans.answer if not ans.error else f"Error: {ans.error}"
        if not status.content and not status.error:
            status.content = "The agent was unable to find an answer"
    except Exception as e:
        logging.exception("Error while processing message: %s", message.content)
        status.content = f"Error: {e}"

    await status.update()

def main():
    print("Use chainlit run src/borrajas/ui/app.py to run the UI")