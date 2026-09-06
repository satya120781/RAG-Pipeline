import json
from urllib import request

from .config import Settings


class OllamaClient:
    """Small HTTP client for Ollama's local embedding and chat endpoints."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.settings.ollama_host}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.settings.request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.settings.ollama_host}. Start it with 'ollama serve'."
            ) from exc

    def embed(self, text: str) -> list[float]:
        result = self._post("/api/embed", {"model": self.settings.embed_model, "input": text})
        embeddings = result.get("embeddings")
        if not embeddings:
            raise RuntimeError("Ollama returned no embeddings")
        return embeddings[0]

    def generate(self, prompt: str) -> str:
        result = self._post(
            "/api/generate",
            {"model": self.settings.llm_model, "prompt": prompt, "stream": False},
        )
        answer = result.get("response")
        if not answer:
            raise RuntimeError("Ollama returned no generated response")
        return answer.strip()
