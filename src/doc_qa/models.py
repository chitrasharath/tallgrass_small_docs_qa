from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Block:
    """A run of cleaned text that belongs to one section of one page."""

    text: str
    section: str | None
    page: int | None
    extraction_method: str  # text | column_repair | ocr | csv_row | csv_note


@dataclass
class PageCheck:
    page: int
    chars: int
    images: int
    two_column: bool = False
    scanned: bool = False
    ocr_confidence: float | None = None


@dataclass
class Document:
    doc_id: str
    source_file: str
    title: str
    doc_date: str | None
    blocks: list[Block]
    page_checks: list[PageCheck] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Passage:
    passage_id: str
    doc_id: str
    source_file: str
    title: str
    page: int | None
    section: str | None
    text: str
    extraction_method: str
    doc_date: str | None
