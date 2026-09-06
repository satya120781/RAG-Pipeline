from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3"
    embed_model: str = "nomic-embed-text"
    request_timeout: float = 120.0
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ollama_host=os.getenv("OLLAMA_HOST", cls.ollama_host).rstrip("/"),
            llm_model=os.getenv("RAG_LLM_MODEL", cls.llm_model),
            embed_model=os.getenv("RAG_EMBED_MODEL", cls.embed_model),
            request_timeout=float(os.getenv("RAG_REQUEST_TIMEOUT", cls.request_timeout)),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", cls.chunk_overlap)),
            top_k=int(os.getenv("RAG_TOP_K", cls.top_k)),
        )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_size must be positive and chunk_overlap must be smaller than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
