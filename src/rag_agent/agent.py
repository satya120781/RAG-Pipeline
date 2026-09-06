from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .documents import chunk_text, load_documents
from .graph import KnowledgeGraph
from .ollama import OllamaClient
from .store import StoredChunk, VectorStore


@dataclass(frozen=True)
class SearchResult:
    source: str
    text: str
    score: float


class RetrievalAgent:
    def __init__(self, db_path: str, settings: Settings | None = None, client=None):
        self.settings = settings or Settings.from_env()
        self.client = client or OllamaClient(self.settings)
        self.store = VectorStore(db_path)
        self.graph = KnowledgeGraph()

    def ingest(self, directory: str | Path, with_graph: bool = False, reset: bool = False) -> int:
        if reset:
            self.store.clear()
            self.graph = KnowledgeGraph()
        total = 0
        for document in load_documents(directory):
            if with_graph:
                self.graph.add_text(document.text, document.source)
            for text in chunk_text(document.text, self.settings.chunk_size, self.settings.chunk_overlap):
                self.store.add(document.source, text, self.client.embed(text))
                total += 1
        return total

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        results = self.store.search(self.client.embed(query), top_k or self.settings.top_k)
        return [SearchResult(item.source, item.text, item.score) for item in results]

    def answer(self, query: str, top_k: int | None = None, with_graph: bool = False) -> tuple[str, list[SearchResult]]:
        results = self.search(query, top_k)
        context = "\n\n".join(f"[{index + 1}] {item.source}\n{item.text}" for index, item in enumerate(results))
        graph_context = self.graph.context(query) if with_graph else ""
        prompt = (
            "You are a careful retrieval agent. Answer only from the supplied context. "
            "If the context is insufficient, say so. Cite supporting chunks as [1], [2], etc.\n\n"
            f"Context:\n{context or '(no matching documents)'}\n\n"
            f"Knowledge graph:\n{graph_context or '(none)'}\n\nQuestion: {query}"
        )
        return self.client.generate(prompt), results

    def close(self) -> None:
        self.store.close()
