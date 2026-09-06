import argparse
import sys

from .agent import RetrievalAgent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local Ollama retrieval agent")
    parser.add_argument("--db", default="./rag.sqlite", help="SQLite index path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("directory")
    ingest.add_argument("--graph", action="store_true")
    ingest.add_argument("--reset", action="store_true")
    ask = subparsers.add_parser("ask")
    ask.add_argument("query")
    ask.add_argument("--top-k", type=int)
    ask.add_argument("--graph", action="store_true")
    ask.add_argument("--no-llm", action="store_true")
    args = parser.parse_args(argv)

    agent = RetrievalAgent(args.db)
    try:
        if args.command == "ingest":
            count = agent.ingest(args.directory, args.graph, args.reset)
            print(f"Indexed {count} chunks.")
        else:
            if args.no_llm:
                for index, result in enumerate(agent.search(args.query, args.top_k), 1):
                    print(f"[{index}] {result.score:.3f} {result.source}\n{result.text}\n")
            else:
                answer, _ = agent.answer(args.query, args.top_k, args.graph)
                print(answer)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        agent.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
