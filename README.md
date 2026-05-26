# Borrajas: A flexible hybrid question-answering system for extreme weather events.

## Overview

Borrajas is a prototype of a question-answering system that can answer questions about extreme weather events
developed within the CLASiK project.
It uses both RAG and RDF sources to answer questions.

## Installation

The project uses `uv` for development, make sure you [install](https://docs.astral.sh/uv/getting-started/installation/) it first.

### Clone the Repository

```bash
git clone https://github.com/your-org/borrajas.git
cd borrajas
```

### Install dependencies

The project supports multiple backends and optional dependencies. You need to install at least one backend.

### LangGraph backend

```bash
uv sync --extra langgraph
```

### Pydantic-AI backend

```bash
uv sync --extra pydantic
```

### ChainLit UI

To use ChainLit UI, install additional dependencies in the `ui` group, e.g.:

```bash
uv sync --extra langgraph --extra ui
```

### All extras

```bash
uv sync --all-extras
```

### Setting up an LLM connection

At the moment, only Ollama API is supported.
Make sure you have Ollama [installed](https://ollama.com/download) and running, 
and you are [signed in](https://docs.ollama.com/cloud#running-cloud-models) to ollama cloud if you are using cloud-based models.

## Usage

There are three ways to run the project:
- `borrajas-cli`: CLI interface, including batch mode
- `borrajas-eval`: Run evaluation on a dataset with a custom config and custom prompts
- `borrajas-ui`: Chatbot with ChainLit UI

### CLI

```bash
# Basic usage
borrajas-cli --backend langgraph --ttl data/sinobas-sample.ttl "How many tornadoes have occurred in the last 10 years in Cantabria?"

# Batch mode
borrajas-cli --backend langgraph --ttl data/sinobas-sample.ttl --questions data/sample-questions.txt

# Using config file
borrajas-cli --config config.yaml --questions data/sample-questions.txt
```

### Evaluation

```bash
borrajas-eval --backend langgraph --config eval/config.yaml
```

### UI

```bash
borrajas-ui --backend langgraph --ttl data/sinobas-sample.ttl
```
