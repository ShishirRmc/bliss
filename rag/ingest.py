"""
Document ingestion and chunking for the RAG pipeline.

Loads markdown files from the knowledge base directory,
splits them into overlapping chunks, and preserves source metadata.
"""

import os
import re
import hashlib


class DocumentIngestor:
    """Load and chunk knowledge base documents."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, kb_dir: str) -> list[dict]:
        """
        Load all .md files from the knowledge base directory.

        Returns:
            List of {"content": str, "source": str, "metadata": dict}
        """
        documents = []
        for filename in sorted(os.listdir(kb_dir)):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(kb_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            documents.append(
                {
                    "content": content,
                    "source": filename,
                    "metadata": {
                        "filepath": filepath,
                        "filename": filename,
                        "char_count": len(content),
                    },
                }
            )
        return documents

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """
        Split documents into overlapping chunks.

        Uses section-aware splitting: tries to split on markdown headers first,
        then falls back to paragraph boundaries, then character boundaries.

        Returns:
            List of {"chunk_id": str, "content": str, "source": str, "metadata": dict}
        """
        all_chunks = []

        for doc in documents:
            sections = self._split_by_sections(doc["content"])

            for section in sections:
                if len(section) <= self.chunk_size:
                    chunk_id = self._make_chunk_id(doc["source"], section)
                    all_chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "content": section.strip(),
                            "source": doc["source"],
                            "metadata": {
                                **doc["metadata"],
                                "chunk_chars": len(section.strip()),
                            },
                        }
                    )
                else:
                    # Further split large sections by paragraphs
                    sub_chunks = self._split_by_size(section)
                    for sub in sub_chunks:
                        chunk_id = self._make_chunk_id(doc["source"], sub)
                        all_chunks.append(
                            {
                                "chunk_id": chunk_id,
                                "content": sub.strip(),
                                "source": doc["source"],
                                "metadata": {
                                    **doc["metadata"],
                                    "chunk_chars": len(sub.strip()),
                                },
                            }
                        )

        # Filter out empty chunks
        all_chunks = [c for c in all_chunks if len(c["content"]) > 20]
        return all_chunks

    def _split_by_sections(self, text: str) -> list[str]:
        """Split markdown text by ## and ### headers."""
        # Split on markdown headers (## or ###)
        pattern = r"(?=^#{2,3}\s)"
        sections = re.split(pattern, text, flags=re.MULTILINE)
        # Keep non-empty sections
        return [s for s in sections if s.strip()]

    def _split_by_size(self, text: str) -> list[str]:
        """Split text into chunks of approximately chunk_size characters."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If single paragraph exceeds chunk_size, split by sentences
                if len(para) > self.chunk_size:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= self.chunk_size:
                            current_chunk += sent + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sent + " "
                else:
                    current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk)

        # Add overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self.chunk_overlap :]
                overlapped.append(prev_tail + chunks[i])
            chunks = overlapped

        return chunks

    def _make_chunk_id(self, source: str, content: str) -> str:
        """Generate a deterministic chunk ID."""
        raw = f"{source}::{content[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
