import argparse
import asyncio
import json
from pathlib import Path

import uvicorn

from modern_rag.config import get_settings
from modern_rag.evaluation import evaluate_retrieval, load_eval_cases
from modern_rag.pipeline import RagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Modern RAG CPU starter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build the local index")
    ingest_parser.add_argument("--source", type=Path, default=None, help="Folder of source documents")

    ask_parser = subparsers.add_parser("ask", help="Ask a grounded question")
    ask_parser.add_argument("query", help="Question to ask")
    ask_parser.add_argument("--top-k", type=int, default=None, help="Override final retrieval count")

    search_parser = subparsers.add_parser("search", help="Retrieve chunks without generation")
    search_parser.add_argument("query", help="Query text")
    search_parser.add_argument("--top-k", type=int, default=None, help="Override final retrieval count")

    eval_parser = subparsers.add_parser("eval", help="Evaluate retrieval hit rate on a JSON dataset")
    eval_parser.add_argument("dataset_path", type=Path, help="Path to evaluation JSON")
    eval_parser.add_argument("--top-k", type=int, default=None, help="Override final retrieval count")

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    settings = get_settings()
    pipeline = RagPipeline(settings)

    if args.command == "ingest":
        result = pipeline.ingest(source=args.source)
        print(json.dumps(result.model_dump(), indent=2))
        return

    if args.command == "search":
        results = pipeline.search(query=args.query, top_k=args.top_k)
        print(json.dumps([result.model_dump() for result in results], indent=2, ensure_ascii=False))
        return

    if args.command == "ask":
        result = asyncio.run(pipeline.ask(query=args.query, top_k=args.top_k))
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        return

    if args.command == "eval":
        cases = load_eval_cases(args.dataset_path)
        result = evaluate_retrieval(pipeline, cases, top_k=args.top_k)
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        return

    if args.command == "serve":
        uvicorn.run("modern_rag.api:app", host=args.host, port=args.port, reload=False)
