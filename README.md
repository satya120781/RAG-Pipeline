# Local RAG Agent

A privacy-first retrieval agent based on the supplied RAG/GraphRAG roadmap. Documents are parsed locally, split into overlapping chunks, embedded through a local Ollama server, and persisted in SQLite. Queries use cosine similarity and can optionally add a lightweight knowledge-graph neighborhood to the generation context.

## User manual

### 1. Requirements

- Python 3.10 or newer.
- [Ollama](https://ollama.com/) installed and available on the local machine.
- Enough disk space for the selected Ollama models and the SQLite index.

The application does not send documents to a hosted service. Ollama runs locally and exposes the model API at `http://localhost:11434` by default.

### 2. Installation

Create and activate a virtual environment, then install the project:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -e ".[test]"
```

Pull one generation model and one embedding model:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

Start Ollama before ingesting or querying:

```bash
ollama serve
```

If Ollama is already running as a desktop service, starting `ollama serve` again is unnecessary.

### 3. Prepare documents

Put documents in a directory, including nested directories if needed:

```text
data/
  handbook.pdf
  architecture.docx
  notes/
    decisions.md
    faq.txt
```

The loader supports:

- `.txt` and `.md` files, read as UTF-8.
- Text-based `.pdf` files, parsed with `pypdf`.
- `.docx` files, parsed with `python-docx`.

Scanned PDFs and images without an extractable text layer are not OCR'd or vision-processed by this PoC.

### 4. Build the index

Run ingestion against the document directory:

```bash
rag-agent --db ./rag.sqlite ingest ./data
```

The command chunks each document, creates an embedding for every chunk through Ollama, and stores the source path, text, and embedding in `rag.sqlite`. The output reports the number of indexed chunks.

To also build the optional in-memory graph context for the current process:

```bash
rag-agent --db ./rag.sqlite ingest ./data --graph
```

Use `--reset` when rebuilding an index from scratch:

```bash
rag-agent --db ./rag.sqlite ingest ./data --reset
```

Without `--reset`, ingesting the same files again adds duplicate chunks. Use `--reset` for repeatable rebuilds.

### 5. Ask questions

Ask a question using the persisted vector index:

```bash
rag-agent --db ./rag.sqlite ask "What is the main conclusion?"
```

The agent retrieves the most similar chunks and asks the local language model to answer only from that context. Retrieved chunks are numbered so the answer can cite them as `[1]`, `[2]`, and so on.

Use graph context when it has been built in the same Python process:

```bash
rag-agent --db ./rag.sqlite ask "How are Ollama and privacy related?" --graph
```

Use `--top-k` to control the number of retrieved chunks:

```bash
rag-agent --db ./rag.sqlite ask "Summarize the deployment approach" --top-k 8
```

To inspect retrieval without calling the generation model:

```bash
rag-agent --db ./rag.sqlite ask "What embedding model is used?" --no-llm
```

### 6. Configuration

Configuration is read from environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `RAG_LLM_MODEL` | `llama3` | Generation model name |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `RAG_REQUEST_TIMEOUT` | `120` | Ollama request timeout in seconds |
| `RAG_CHUNK_SIZE` | `1200` | Approximate chunk size in characters |
| `RAG_CHUNK_OVERLAP` | `200` | Approximate overlap between chunks |
| `RAG_TOP_K` | `5` | Default number of retrieved chunks |

Example:

```bash
# Windows PowerShell
$env:RAG_LLM_MODEL = "mistral"
$env:RAG_TOP_K = "8"

# macOS/Linux
export RAG_LLM_MODEL=mistral
export RAG_TOP_K=8
```

The embedding model must remain the same for ingestion and querying. If it changes, rebuild the database with `--reset`.

### 7. Python API

The CLI is the recommended interface, but the agent can also be embedded in another Python application:

```python
from rag_agent import RetrievalAgent

agent = RetrievalAgent("./rag.sqlite")
try:
    agent.ingest("./data", reset=True)
    answer, sources = agent.answer("What is the main conclusion?")
    print(answer)
    for source in sources:
        print(source.source, source.score)
finally:
    agent.close()
```

Pass a custom `Settings` instance and Ollama-compatible client when integrating with another application or writing tests.

### 8. Testing

The test suite uses fake embeddings and generation, so it runs offline and does not require Ollama:

```bash
pytest
```

The tests cover chunking, cosine similarity validation, indexing, retrieval, and answer generation.

### 9. Troubleshooting

**`Could not reach Ollama`**: Start Ollama with `ollama serve`, verify the configured `OLLAMA_HOST`, and confirm the required models exist with `ollama list`.

**`Ollama returned no embeddings`**: Check that the embedding model was pulled and that `RAG_EMBED_MODEL` matches its installed name.

**Poor retrieval results**: Use a model suited to your language, adjust `RAG_CHUNK_SIZE` and `RAG_CHUNK_OVERLAP`, and rebuild the database after changing the embedding model.

**No text from a PDF**: The current parser handles text-based PDFs only. Run OCR separately before ingestion for scanned documents.

## Quick start

For a minimal end-to-end run:

```bash
pip install -e ".[test]"
ollama serve
ollama pull llama3
ollama pull nomic-embed-text
rag-agent --db ./rag.sqlite ingest ./data --reset
rag-agent --db ./rag.sqlite ask "What is the main conclusion?"
```
