import json
import math
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredChunk:
    id: int
    source: str
    text: str
    score: float


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("vectors must have the same non-zero dimensionality")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


class VectorStore:
    def __init__(self, path: str):
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, source TEXT, text TEXT, embedding TEXT NOT NULL)"
        )
        self.connection.commit()

    def add(self, source: str, text: str, embedding: list[float]) -> None:
        self.connection.execute(
            "INSERT INTO chunks(source, text, embedding) VALUES (?, ?, ?)",
            (source, text, json.dumps(embedding)),
        )
        self.connection.commit()

    def clear(self) -> None:
        self.connection.execute("DELETE FROM chunks")
        self.connection.commit()

    def search(self, embedding: list[float], top_k: int) -> list[StoredChunk]:
        rows = self.connection.execute("SELECT id, source, text, embedding FROM chunks").fetchall()
        scored = [
            StoredChunk(row[0], row[1], row[2], cosine_similarity(embedding, json.loads(row[3])))
            for row in rows
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def count(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self.connection.close()
