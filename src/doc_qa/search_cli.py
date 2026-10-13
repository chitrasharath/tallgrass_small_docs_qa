"""Search the prepared passages.

Usage: uv run doc-qa-search "How much is mileage?" [--method all|bm25|local-embedding|api-embedding] [-k 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from doc_qa.embedders import EmbeddingError
from doc_qa.search import DEFAULT_PASSAGES, SEARCH_METHODS, load_passages, make_searcher, resolve_methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--method", choices=["all", *SEARCH_METHODS], default="all")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES)
    args = parser.parse_args()

    passages = load_passages(args.passages)
    for method in resolve_methods(args.method):
        print(f"\n== {method} ==")
        try:
            hits = make_searcher(method, passages).search(args.query, args.k)
        except EmbeddingError as exc:
            parser.exit(1, f"error: {exc}\n")
        for hit in hits:
            p = hit.passage
            page = f" p{p.page}" if p.page else ""
            print(f"{hit.rank}. [{hit.score:.3f}] {p.passage_id} ({p.source_file}{page}, {p.section})")
            print(f"   {p.text[:160].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
