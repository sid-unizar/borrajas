# Borrajas: A flexible hybrid question-answering system for extreme weather events.

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

### Setting up LLM connection

At the moment, only Ollama API is supported.
Make sure you have Ollama [installed](https://ollama.com/download) and running, 
and you are [signed in](https://docs.ollama.com/cloud#running-cloud-models) to ollama cloud if you are using cloud-based models.