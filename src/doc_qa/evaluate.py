"""Score the search step against the evaluation questions.

A passage is relevant to a question when it comes from the expected source file and
contains every evidence phrase. Questions with no source (no answer in the corpus)
are listed but not scored.

Metrics, averaged over the scored questions:
- Recall@k: share of a question's relevant passages that appear in the top k.
- Precision@k: share of the top k that are relevant.
- MRR: mean of 1 / rank of the first relevant passage (0 when none is found).

Usage: uv run doc-qa-eval [--method all|bm25|local-embedding|api-embedding] [-k 1 3 5] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from doc_qa.embedders import EmbeddingError
from doc_qa.models import Passage
from doc_qa.search import (
    DEFAULT_PASSAGES,
    SEARCH_METHODS,
    EmbeddingSearch,
    Searcher,
    load_passages,
    make_searcher,
    resolve_methods,
)

DEFAULT_QUESTIONS = Path("memory_bank/references/questions.jsonl")


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_relevant(question: dict, passage: Passage) -> bool:
    return passage.source_file == question["source"] and all(
        e in passage.text for e in question["evidence"]
    )


@dataclass
class QuestionResult:
    question: dict
    ranked_ids: list[str]  # top max(k) passage ids
    relevant_ids: set[str]
    top_score: float | None

    @property
    def first_relevant_rank(self) -> int | None:
        for rank, passage_id in enumerate(self.ranked_ids, start=1):
            if passage_id in self.relevant_ids:
                return rank
        return None

    def recall(self, k: int) -> float:
        found = len(self.relevant_ids & set(self.ranked_ids[:k]))
        return found / len(self.relevant_ids) if self.relevant_ids else 0.0

    def precision(self, k: int) -> float:
        return len(self.relevant_ids & set(self.ranked_ids[:k])) / k

    @property
    def reciprocal_rank(self) -> float:
        rank = self.first_relevant_rank
        return 1 / rank if rank else 0.0


@dataclass
class MethodResult:
    name: str
    label: str
    ks: list[int]
    results: list[QuestionResult]

    @property
    def scored(self) -> list[QuestionResult]:
        return [r for r in self.results if r.question["source"] is not None]

    def recall(self, k: int, kind: str | None = None) -> float:
        rows = [r for r in self.scored if kind is None or r.question["kind"] == kind]
        return sum(r.recall(k) for r in rows) / len(rows) if rows else 0.0

    def precision(self, k: int) -> float:
        return sum(r.precision(k) for r in self.scored) / len(self.scored) if self.scored else 0.0

    @property
    def mrr(self) -> float:
        return sum(r.reciprocal_rank for r in self.scored) / len(self.scored) if self.scored else 0.0


def evaluate(
    searcher: Searcher, passages: list[Passage], questions: list[dict], ks: list[int]
) -> MethodResult:
    depth = max(ks)
    if isinstance(searcher, EmbeddingSearch):
        searcher.prefetch([q["question"] for q in questions])
    results = []
    for question in questions:
        hits = searcher.search(question["question"], depth)
        relevant = (
            {p.passage_id for p in passages if is_relevant(question, p)}
            if question["source"]
            else set()
        )
        results.append(
            QuestionResult(
                question=question,
                ranked_ids=[h.passage.passage_id for h in hits],
                relevant_ids=relevant,
                top_score=hits[0].score if hits else None,
            )
        )
    label = (
        f"Embedding ({searcher.embedder.model_name})"
        if isinstance(searcher, EmbeddingSearch)
        else "BM25"
    )
    return MethodResult(name=searcher.name, label=label, ks=ks, results=results)


def print_summary(result: MethodResult) -> None:
    print(f"\n== {result.name} ==")
    for r in result.results:
        rank = r.first_relevant_rank
        if r.question["source"] is None:
            status = f"not scored (top score {r.top_score:.3f})"
        else:
            status = f"rank {rank}" if rank else f"MISS in top {max(result.ks)}"
        print(f"{r.question['id']}  {r.question['kind']:<10} {status}")
    for k in result.ks:
        print(f"Recall@{k} {result.recall(k):.3f}  Precision@{k} {result.precision(k):.3f}")
    print(f"MRR {result.mrr:.3f}")


def render_report(
    results: list[MethodResult], questions_path: Path, passages_path: Path, method_arg: str
) -> str:
    ks = results[0].ks
    scored = results[0].scored
    unscored = [r for r in results[0].results if r.question["source"] is None]
    relevant_counts = sorted({len(r.relevant_ids) for r in scored})
    kinds = list(dict.fromkeys(r.question["kind"] for r in scored))
    passage_count = len(load_passages(passages_path))

    lines = [
        f"# Search results: {' vs '.join(r.label for r in results)}",
        "",
        f"Run on {date.today().isoformat()} with "
        f"`uv run doc-qa-eval --method {method_arg} -k {' '.join(map(str, ks))} --report <this file>`.",
        "",
        f"- Questions: `{questions_path}` ({len(scored)} scored, {len(unscored)} with no answer in the corpus, not scored)",
        f"- Passages: `{passages_path}` ({passage_count} passages)",
        "- A passage is relevant when it comes from the expected source file and contains every evidence phrase.",
        f"- Relevant passages per question: {', '.join(map(str, relevant_counts))}. "
        + (
            f"With one relevant passage per question, Recall@k equals the hit rate and the best possible Precision@k is 1/k "
            f"({', '.join(f'{1 / k:.3f} at k={k}' for k in ks)})."
            if relevant_counts == [1]
            else "Precision@k is capped by how many relevant passages each question has."
        ),
        "",
        "## Summary",
        "",
        "| Method | " + " | ".join(f"Recall@{k}" for k in ks) + " | "
        + " | ".join(f"Precision@{k}" for k in ks) + " | MRR |",
        "|---|" + "---|" * (2 * len(ks) + 1),
    ]
    for result in results:
        cells = [f"{result.recall(k):.3f}" for k in ks] + [f"{result.precision(k):.3f}" for k in ks]
        lines.append(f"| {result.label} | " + " | ".join(cells) + f" | {result.mrr:.3f} |")

    lines += [
        "",
        "## Recall by question kind",
        "",
        "| Method | Kind | " + " | ".join(f"Recall@{k}" for k in ks) + " |",
        "|---|---|" + "---|" * len(ks),
    ]
    for result in results:
        for kind in kinds:
            count = sum(1 for r in result.scored if r.question["kind"] == kind)
            cells = [f"{result.recall(k, kind):.3f}" for k in ks]
            lines.append(f"| {result.label} | {kind} ({count}) | " + " | ".join(cells) + " |")

    depth = max(ks)
    lines += [
        "",
        "## Per question",
        "",
        f"Rank of the first relevant passage in the top {depth} (\"miss\" when it is not there).",
        "",
        "| Id | Kind | Question | Relevant passage | " + " | ".join(r.label for r in results) + " |",
        "|---|---|---|---|" + "---|" * len(results),
    ]
    for index, row in enumerate(results[0].results):
        question = row.question
        if question["source"] is None:
            ranks = [f"not scored (top score {r.results[index].top_score:.3f})" for r in results]
            relevant = "none"
        else:
            ranks = [str(r.results[index].first_relevant_rank or "miss") for r in results]
            relevant = ", ".join(f"`{i}`" for i in sorted(row.relevant_ids))
        lines.append(
            f"| {question['id']} | {question['kind']} | {question['question']} | {relevant} | "
            + " | ".join(ranks)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=["all", *SEARCH_METHODS], default="all")
    parser.add_argument("-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--report", type=Path, help="write a Markdown report to this path")
    args = parser.parse_args()

    passages = load_passages(args.passages)
    questions = load_questions(args.questions)
    ks = sorted(set(args.k))
    methods = resolve_methods(args.method)
    try:
        results = [evaluate(make_searcher(m, passages), passages, questions, ks) for m in methods]
    except EmbeddingError as exc:
        parser.exit(1, f"error: {exc}\n")
    for result in results:
        print_summary(result)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(results, args.questions, args.passages, args.method), encoding="utf-8")
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
