from __future__ import annotations

import csv
from pathlib import Path

from doc_qa.models import Block, Document
from doc_qa.normalize import find_doc_date, normalize_text


def _non_empty(row: list[str]) -> list[str]:
    return [cell for cell in row if cell]


def read_table(path: Path) -> Document:
    """Read a CSV that may have title lines above the header and notes below the table.

    The header is the first row with every cell filled. Data rows follow until the
    first row with fewer than two filled cells; any text after that is kept as notes.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [[normalize_text(cell) for cell in row] for row in csv.reader(handle)]

    header_index = next(
        (i for i, row in enumerate(rows) if len(row) >= 2 and all(row)), None
    )
    if header_index is None:
        raise ValueError(f"{path.name}: no header row with every cell filled")
    header = rows[header_index]

    preamble = [" ".join(_non_empty(row)) for row in rows[:header_index] if _non_empty(row)]
    title = preamble[0] if preamble else path.stem
    context = " ".join(preamble[1:])

    data: list[list[str]] = []
    position = header_index + 1
    while position < len(rows) and len(_non_empty(rows[position])) >= 2:
        data.append(rows[position])
        position += 1
    notes = [" ".join(_non_empty(row)) for row in rows[position:] if _non_empty(row)]

    lead = f"{title}. {context}".strip()
    blocks: list[Block] = []
    if context:
        blocks.append(Block(text=lead, section="About this table", page=None, extraction_method="csv_note"))
    for row in data:
        fields = "; ".join(f"{name}: {value}" for name, value in zip(header, row) if value)
        blocks.append(
            Block(
                text=f"{lead}\n{fields}",
                section=" - ".join(_non_empty(row)[:2]),
                page=None,
                extraction_method="csv_row",
            )
        )
    for note in notes:
        blocks.append(Block(text=f"{title}. {note}", section="Notes", page=None, extraction_method="csv_note"))

    return Document(
        doc_id=path.stem,
        source_file=path.name,
        title=title,
        doc_date=find_doc_date("\n".join(preamble + notes)),
        blocks=blocks,
        metadata={"columns": ", ".join(header), "rows": str(len(data))},
    )
