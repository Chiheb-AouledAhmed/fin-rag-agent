"""Loading and chunking of the document corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a document. `id` is what the model cites, e.g. `basel3#2`."""

    id: str
    doc: str
    title: str
    section: str
    text: str


_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")


def load_markdown(path: Path, max_chars: int = 900) -> list[Chunk]:
    """Split a markdown file into chunks along `##` sections, then by paragraph if a section is long.

    Section-aware chunking keeps each chunk on a single topic, which makes citations meaningful
    and retrieval more precise than fixed-size windows.
    """
    doc = path.stem
    title = doc
    section = ""
    sections: list[tuple[str, list[str]]] = []
    buf: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        m = _HEADING.match(line)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1:
                title = text
                continue
            if buf:
                sections.append((section, buf))
            section, buf = text, []
        elif line.strip():
            buf.append(line.strip())
    if buf:
        sections.append((section, buf))

    chunks: list[Chunk] = []
    for sec, paragraphs in sections:
        current = ""
        for p in paragraphs:
            if current and len(current) + len(p) + 1 > max_chars:
                chunks.append(Chunk(f"{doc}#{len(chunks)}", doc, title, sec, current))
                current = p
            else:
                current = f"{current} {p}".strip()
        if current:
            chunks.append(Chunk(f"{doc}#{len(chunks)}", doc, title, sec, current))
    return chunks


def load_corpus(directory: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(load_markdown(path))
    if not chunks:
        raise FileNotFoundError(f"no markdown documents found in {directory}")
    return chunks
