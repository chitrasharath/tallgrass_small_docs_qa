from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page

from doc_qa.models import Block, Document, PageCheck
from doc_qa.normalize import find_doc_date, join_lines, normalize_text
from doc_qa.ocr import fix_ocr_text, merge_broken_paragraphs, ocr_image

SCANNED_MAX_CHARS = 20
SEGMENT_GAP = 12.0  # points; word gaps are ~3, a column gutter is much wider
LINE_TOLERANCE = 3.0  # points of vertical drift allowed within one line
MIN_SPLIT_LINES = 5
MIN_SPLIT_RATIO = 0.5
INDENT = 4.0
OCR_RESOLUTION = 300
_FIELD = re.compile(r"^(To|From|Date|Re|Subject|Cc)\s*:\s*(.+)$", re.IGNORECASE)


@dataclass
class Line:
    words: list[dict]

    @property
    def x0(self) -> float:
        return self.words[0]["x0"]

    @property
    def x1(self) -> float:
        return self.words[-1]["x1"]

    @property
    def top(self) -> float:
        return min(w["top"] for w in self.words)

    @property
    def text(self) -> str:
        return " ".join(w["text"] for w in self.words)

    @property
    def size(self) -> float:
        return max(round(w["size"], 1) for w in self.words)

    @property
    def bold(self) -> bool:
        return all("bold" in w["fontname"].lower() for w in self.words)


def _group_lines(words: list[dict]) -> list[Line]:
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(word["top"] - lines[-1][0]["top"]) <= LINE_TOLERANCE:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [Line(sorted(line, key=lambda w: w["x0"])) for line in lines]


def _segments(line: Line) -> list[Line]:
    segments = [[line.words[0]]]
    for previous, word in zip(line.words, line.words[1:]):
        if word["x0"] - previous["x1"] > SEGMENT_GAP:
            segments.append([word])
        else:
            segments[-1].append(word)
    return [Line(segment) for segment in segments]


def _gutter_split(line: Line, center: float) -> tuple[Line, Line] | None:
    """Left and right parts of a line whose wide gap covers the page center."""
    segments = _segments(line)
    for left, right in zip(segments, segments[1:]):
        if left.x1 < center < right.x0:
            left_words = [w for w in line.words if w["x1"] <= left.x1]
            right_words = [w for w in line.words if w["x0"] >= right.x0]
            return Line(left_words), Line(right_words)
    return None


def find_gutter(lines: list[Line], page_width: float) -> float | None:
    """x position of the column gutter, or None when the page reads as one column."""
    center = page_width / 2
    splits = [s for s in (_gutter_split(line, center) for line in lines) if s]
    if len(splits) < MIN_SPLIT_LINES or len(splits) < MIN_SPLIT_RATIO * len(lines):
        return None
    left_edge = max(left.x1 for left, _ in splits)
    right_edge = min(right.x0 for _, right in splits)
    return (left_edge + right_edge) / 2


def _reading_order(lines: list[Line], gutter: float) -> list[list[Line]]:
    """Full-width header lines, left column, right column, full-width footer lines."""
    split_rows = [i for i, line in enumerate(lines) if _gutter_split(line, gutter)]
    first, last = split_rows[0], split_rows[-1]
    header, footer = lines[:first], lines[last + 1 :]
    left_words, right_words = [], []
    for line in lines[first : last + 1]:
        for word in line.words:
            (left_words if word["x0"] < gutter else right_words).append(word)
    return [header, _group_lines(left_words), _group_lines(right_words), footer]


def _body_size(lines: list[Line]) -> float:
    sizes = Counter(round(w["size"], 1) for line in lines for w in line.words)
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _blocks_from_lines(
    regions: list[list[Line]],
    page_number: int,
    method: str,
    title: str,
    reset_regions: frozenset[int] = frozenset(),
) -> list[Block]:
    """Rebuild paragraphs and sections from lines that are already in reading order.

    Regions listed in reset_regions start over under the document title instead of
    continuing the previous region's section.
    """
    all_lines = [line for region in regions for line in region]
    body = _body_size(all_lines)
    spacings = [
        b.top - a.top
        for region in regions
        for a, b in zip(region, region[1:])
        if a.size == body and b.size == body
    ]
    spacing = statistics.median(spacings) if spacings else body * 1.3

    blocks: list[Block] = []
    section = title
    paragraphs: list[str] = []

    def flush() -> None:
        if paragraphs:
            blocks.append(
                Block(
                    text="\n\n".join(paragraphs),
                    section=section,
                    page=page_number,
                    extraction_method=method,
                )
            )
            paragraphs.clear()

    for index, region in enumerate(regions):
        if not region:
            continue
        if index in reset_regions:
            flush()
            section = title
        left_margin = min(line.x0 for line in region)
        current: list[str] = []
        previous: Line | None = None
        for line in region:
            if line.bold and line.size > body:
                if current:
                    paragraphs.append(join_lines(current))
                    current = []
                flush()
                section = line.text
                previous = None
                continue
            new_paragraph = previous is None or (
                line.x0 > left_margin + INDENT or line.top - previous.top > spacing * 1.5
            )
            if new_paragraph and current:
                paragraphs.append(join_lines(current))
                current = []
            current.append(line.text)
            previous = line
        if current:
            paragraphs.append(join_lines(current))
        flush()
    return blocks


def _text_page_blocks(page: Page, check: PageCheck, title: str) -> list[Block]:
    words = page.extract_words(extra_attrs=["fontname", "size"])
    if not words:
        return []
    lines = _group_lines(words)
    gutter = find_gutter(lines, float(page.width))
    if gutter is None:
        return _blocks_from_lines([lines], check.page, "text", title)
    check.two_column = True
    regions = _reading_order(lines, gutter)
    return _blocks_from_lines(
        regions, check.page, "column_repair", title, reset_regions=frozenset({len(regions) - 1})
    )


def _scanned_page_blocks(page: Page, check: PageCheck, metadata: dict[str, str]) -> list[Block]:
    result = ocr_image(page.to_image(resolution=OCR_RESOLUTION).original)
    check.ocr_confidence = result.confidence
    paragraphs: list[str] = []
    for lines in result.paragraphs:
        lines = [fix_ocr_text(normalize_text(line)) for line in lines]
        fields = [_FIELD.match(line) for line in lines]
        if all(fields):
            for match in fields:
                metadata.setdefault(match.group(1).lower(), match.group(2))
            paragraphs.append("\n".join(lines))
        else:
            paragraphs.append(join_lines(lines))
    paragraphs = merge_broken_paragraphs(paragraphs)
    if not paragraphs:
        return []
    section = metadata.get("re") or metadata.get("subject")
    return [Block(text="\n\n".join(paragraphs), section=section, page=check.page, extraction_method="ocr")]


def _pdf_title(pdf: pdfplumber.PDF, path: Path) -> str | None:
    title = (pdf.metadata or {}).get("Title")
    if isinstance(title, str) and title.strip() and title.strip() != path.stem:
        return normalize_text(title)
    return None


def read_pdf(path: Path) -> Document:
    metadata: dict[str, str] = {}
    blocks: list[Block] = []
    checks: list[PageCheck] = []
    with pdfplumber.open(path) as pdf:
        title = _pdf_title(pdf, path)
        for number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            check = PageCheck(page=number, chars=len(text.strip()), images=len(page.images))
            check.scanned = check.chars < SCANNED_MAX_CHARS and check.images > 0
            checks.append(check)
            if check.scanned:
                blocks.extend(_scanned_page_blocks(page, check, metadata))
            else:
                blocks.extend(_text_page_blocks(page, check, title or path.stem))

    for block in blocks:
        block.text = normalize_text(block.text)
        block.section = normalize_text(block.section) if block.section else None

    full_text = "\n".join(block.text for block in blocks)
    if title is None:
        subject = metadata.get("re") or metadata.get("subject")
        title = subject or (full_text.splitlines()[0] if full_text else path.stem)
    return Document(
        doc_id=path.stem,
        source_file=path.name,
        title=title,
        doc_date=metadata.get("date") or find_doc_date(full_text),
        blocks=blocks,
        page_checks=checks,
        metadata=metadata,
    )
