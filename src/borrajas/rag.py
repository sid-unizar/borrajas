import hashlib
import logging
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Config

CHROMA_ROOT = "data/rag/chroma"


def _read_urls_file(path: str | Path) -> list[str]:
    path = Path(path)

    if not path.exists():
        logging.warning("RAG URLs file does not exist: %s", path.absolute())
        return []

    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        urls.append(line)

    return urls


def _get_rag_urls(config: Config) -> list[str]:
    urls = []

    urls_path = config.params.get("rag_urls_path")
    if urls_path:
        urls.extend(_read_urls_file(urls_path))

    inline_urls = config.params.get("rag_urls")
    if isinstance(inline_urls, str):
        urls.extend(url.strip() for url in inline_urls.split(",") if url.strip())
    elif isinstance(inline_urls, list):
        urls.extend(str(url).strip() for url in inline_urls if str(url).strip())

    if not urls:
        logging.warning("No RAG URLs provided")
        return []

    return list(dict.fromkeys(urls))


def _load_url(url: str) -> Document:
    response = requests.get(
        url,
        headers={"User-Agent": "borrajas-rag/0.1"},
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    text = "\n".join(
        element.get_text(" ", strip=True)
        for element in soup.find_all(["h1", "h2", "h3", "p", "li"])
        if len(element.get_text(" ", strip=True)) > 30
    )

    return Document(
        page_content=f"Title: {title}\nURL: {url}\n\n{text}",
        metadata={
            "source": url,
            "title": title,
        },
    )


def _source_signature(urls: list[str]) -> str:
    return hashlib.sha1("\n".join(sorted(urls)).encode("utf-8")).hexdigest()[:12]


def init_rag(config: Config) -> Optional[object]:
    urls = _get_rag_urls(config)

    if not urls:
        logging.warning("RAG enabled but no URLs configured")
        return None

    embeddings = OllamaEmbeddings(
        model=config.params.get("embedding_model", "nomic-embed-text")
    )

    chroma_dir = Path(CHROMA_ROOT) / _source_signature(urls)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    if any(chroma_dir.iterdir()):
        logging.info("Loading existing RAG index from %s", chroma_dir)
        return Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embeddings,
        ).as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

    logging.info("Building RAG index from %d URL(s)", len(urls))

    documents = []
    for url in urls:
        try:
            documents.append(_load_url(url))
            logging.info("Loaded RAG URL: %s", url)
        except Exception as exc:
            logging.warning("Could not load RAG URL %s: %s", url, exc)

    if not documents:
        logging.warning("No RAG documents loaded")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
    )
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(chroma_dir),
    )

    logging.info("Built RAG index with %d chunks", len(chunks))

    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})