from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    source: str


class KnowledgeGraph:
    """Deterministic local graph fallback; it requires no database or remote service."""

    def __init__(self):
        self.relations: list[Relation] = []

    def add_text(self, text: str, source: str) -> None:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            entities = re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b", sentence)
            for subject, obj in zip(entities, entities[1:]):
                self.relations.append(Relation(subject, "RELATED_TO", obj, source))

    def context(self, query: str, limit: int = 10) -> str:
        terms = {term.lower() for term in re.findall(r"\w+", query)}
        matches = [
            relation for relation in self.relations
            if relation.subject.lower() in terms or relation.object.lower() in terms
        ][:limit]
        return "\n".join(
            f"{relation.subject} -[{relation.predicate}]-> {relation.object} ({relation.source})"
            for relation in matches
        )
