from pathlib import Path

from rag_agent.agent import RetrievalAgent
from rag_agent.config import Settings
from rag_agent.documents import chunk_text
from rag_agent.store import cosine_similarity


class FakeOllama:
    def embed(self, text):
        return [float(text.lower().count(word)) for word in ("ollama", "graph", "privacy", "python")]

    def generate(self, prompt):
        assert "Answer only from the supplied context" in prompt
        return "The answer is supported by the retrieved context. [1]"


def settings():
    return Settings(chunk_size=80, chunk_overlap=10, top_k=2)


def test_chunking_overlaps_and_preserves_text():
    chunks = chunk_text("one two three four five six seven eight", size=15, overlap=5)
    assert len(chunks) > 1
    assert "three" in chunks[0]
    assert chunks[1]


def test_cosine_similarity_rejects_invalid_vectors():
    assert cosine_similarity([1, 0], [1, 0]) == 1
    try:
        cosine_similarity([1], [1, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected dimensionality validation")


def test_ingest_search_and_answer_are_offline(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "notes.txt").write_text(
        "Ollama enables privacy-first Python retrieval. Graph context connects concepts.",
        encoding="utf-8",
    )
    agent = RetrievalAgent(str(tmp_path / "index.sqlite"), settings(), FakeOllama())
    try:
        assert agent.ingest(data, with_graph=True) > 0
        results = agent.search("privacy Ollama")
        assert results and results[0].source.endswith("notes.txt")
        answer, cited = agent.answer("How does Ollama help privacy?")
        assert cited
        assert "[1]" in answer
    finally:
        agent.close()
