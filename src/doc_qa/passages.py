from __future__ import annotations

import re

from doc_qa.models import Document, Passage

MAX_WORDS = 150
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$\"'(])")


def _words(text: str) -> int:
    return len(text.split())


def _split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    """Pack whole sentences into chunks, repeating the last sentence at the start of the next."""
    sentences = _SENTENCE_END.split(paragraph)
    chunks: list[list[str]] = [[]]
    for sentence in sentences:
        if chunks[-1] and _words(" ".join(chunks[-1] + [sentence])) > max_words:
            chunks.append([chunks[-1][-1]])
        chunks[-1].append(sentence)
    return [" ".join(chunk) for chunk in chunks]


def split_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Group paragraphs into passages of at most max_words, never cutting a sentence."""
    pieces: list[str] = []
    for paragraph in text.split("\n\n"):
        if _words(paragraph) > max_words:
            pieces.extend(_split_long_paragraph(paragraph, max_words))
        else:
            pieces.append(paragraph)

    passages: list[str] = []
    for piece in pieces:
        if passages and _words(passages[-1]) + _words(piece) <= max_words:
            passages[-1] += "\n\n" + piece
        else:
            passages.append(piece)
    return passages


def build_passages(document: Document, max_words: int = MAX_WORDS) -> list[Passage]:
    passages: list[Passage] = []
    for block in document.blocks:
        for text in split_text(block.text, max_words):
            passages.append(
                Passage(
                    passage_id=f"{document.doc_id}-{len(passages) + 1:03d}",
                    doc_id=document.doc_id,
                    source_file=document.source_file,
                    title=document.title,
                    page=block.page,
                    section=block.section,
                    text=text,
                    extraction_method=block.extraction_method,
                    doc_date=document.doc_date,
                )
            )
    return passages
