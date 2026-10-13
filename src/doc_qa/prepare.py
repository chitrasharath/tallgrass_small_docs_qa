"""Read every document in the corpus, clean it, and write passages for indexing.

Usage: uv run doc-qa-prepare [corpus_dir] [out_dir]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from doc_qa.markdown_reader import read_markdown
from doc_qa.models import Document
from doc_qa.passages import build_passages
from doc_qa.pdf_reader import read_pdf
from doc_qa.table_reader import read_table

READERS: dict[str, Callable[[Path], Document]] = {
    ".md": read_markdown,
    ".markdown": read_markdown,
    ".csv": read_table,
    ".pdf": read_pdf,
}


def read_document(path: Path) -> Document:
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"No reader for {path.name}")
    return reader(path)


def render_cleaned(document: Document) -> str:
    methods = sorted({block.extraction_method for block in document.blocks})
    lines = [
        f"# {document.title}",
        "",
        f"source_file: {document.source_file}",
        f"doc_date: {document.doc_date or 'unknown'}",
        f"extraction: {', '.join(methods) or 'none'}",
    ]
    lines += [f"{key}: {value}" for key, value in document.metadata.items()]
    for block in document.blocks:
        page = f" (page {block.page})" if block.page else ""
        lines += ["", f"## {block.section or document.title}{page}", "", block.text]
    return "\n".join(lines) + "\n"


def prepare(corpus_dir: Path, out_dir: Path) -> list[Document]:
    out_dir.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []
    passage_rows: list[str] = []
    for path in sorted(p for p in corpus_dir.iterdir() if p.suffix.lower() in READERS):
        document = read_document(path)
        documents.append(document)
        (out_dir / f"{document.doc_id}.md").write_text(render_cleaned(document), encoding="utf-8")
        passages = build_passages(document)
        passage_rows += [json.dumps(asdict(p), ensure_ascii=False) for p in passages]
        _report(document, len(passages))

    (out_dir / "passages.jsonl").write_text("\n".join(passage_rows) + "\n", encoding="utf-8")
    manifest = [
        {
            "doc_id": d.doc_id,
            "source_file": d.source_file,
            "title": d.title,
            "doc_date": d.doc_date,
            "metadata": d.metadata,
            "page_checks": [asdict(c) for c in d.page_checks],
        }
        for d in documents
    ]
    (out_dir / "documents.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(documents)} documents, {len(passage_rows)} passages -> {out_dir}")
    return documents


def _report(document: Document, passage_count: int) -> None:
    flags = []
    for check in document.page_checks:
        if check.two_column:
            flags.append(f"p{check.page} two-column (repaired)")
        if check.scanned:
            flags.append(f"p{check.page} scanned (OCR confidence {check.ocr_confidence})")
    note = f"  [{'; '.join(flags)}]" if flags else ""
    print(f"{document.source_file:<36} {len(document.blocks):>3} blocks {passage_count:>3} passages{note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus_dir", nargs="?", default="data/corpus", type=Path)
    parser.add_argument("out_dir", nargs="?", default="data/processed", type=Path)
    args = parser.parse_args()
    prepare(args.corpus_dir, args.out_dir)


if __name__ == "__main__":
    main()
