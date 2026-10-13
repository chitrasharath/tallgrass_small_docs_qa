from __future__ import annotations

import re
from pathlib import Path

from doc_qa.models import Block, Document
from doc_qa.normalize import find_doc_date, join_lines, normalize_text

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_PAGE_FOOTER = re.compile(r"^.{0,100}\bPage\s+(\d+)\s+of\s+\d+\s*$", re.IGNORECASE)
_SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
_INLINE = [
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"),
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"), r"\1"),
    (re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)"), r"\1"),
    (re.compile(r"`([^`]+)`"), r"\1"),
]


def _strip_inline(text: str) -> str:
    for pattern, repl in _INLINE:
        text = pattern.sub(repl, text)
    return text


def _clean_heading(text: str) -> str:
    return _SECTION_NUMBER.sub("", _strip_inline(text)).strip()


def read_markdown(path: Path) -> Document:
    lines = normalize_text(path.read_text(encoding="utf-8")).splitlines()
    has_page_footers = any(_PAGE_FOOTER.match(line) for line in lines)

    title = path.stem
    page = 1
    headings: list[tuple[int, str]] = []
    blocks: list[Block] = []
    paragraphs: list[str] = []
    current: list[str] = []

    def flush_paragraph() -> None:
        if current:
            paragraphs.append(_strip_inline(join_lines(current)))
            current.clear()

    def flush_block() -> None:
        flush_paragraph()
        if paragraphs:
            section = " > ".join(name for _, name in headings) or title
            blocks.append(
                Block(
                    text="\n\n".join(paragraphs),
                    section=section,
                    page=page if has_page_footers else None,
                    extraction_method="text",
                )
            )
            paragraphs.clear()

    seen_title = False
    for line in lines:
        if footer := _PAGE_FOOTER.match(line):
            flush_block()
            page = int(footer.group(1)) + 1
            continue
        if heading := _HEADING.match(line):
            flush_block()
            level, name = len(heading.group(1)), _clean_heading(heading.group(2))
            if level == 1 and not seen_title:
                title, seen_title = name, True
                continue
            headings = [(lvl, n) for lvl, n in headings if lvl < level]
            headings.append((level, name))
            continue
        if not line:
            flush_paragraph()
            continue
        current.append(line)
    flush_block()

    return Document(
        doc_id=path.stem,
        source_file=path.name,
        title=title,
        doc_date=find_doc_date("\n".join(lines)),
        blocks=blocks,
    )
