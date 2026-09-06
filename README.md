# Local RAG Agent

This is a privacy-first retrieval agent based on the attached RAG/GraphRAG roadmap. Documents are parsed locally, split into overlapping chunks, embedded through a local Ollama server, and persisted in SQLite. Queries use cosine similarity and can optionally add a lightweight knowledge-graph neighborhood to the generation context.

## Quick start

Install the package and pull the two local Ollama models:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[test]"
ollama pull llama3
ollama pull nomic-embed-text
```

Ingest a directory containing `.txt`, `.md`, `.pdf`, or `.docx` files, then ask a question:

```bash
rag-agent --db ./rag.sqlite ingest ./data --graph
rag-agent --db ./rag.sqlite ask "What is the main conclusion?" --graph
```

Ollama must be running locally (`ollama serve`). The default endpoint is `http://localhost:11434`; configure it with `OLLAMA_HOST`, `RAG_LLM_MODEL`, and `RAG_EMBED_MODEL`. Use `--no-llm` to inspect retrieved context without generation.

The test suite uses fake embeddings and generation, so `pytest` runs offline and does not require Ollama.
