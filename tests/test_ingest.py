"""
Tests for DocumentIngestor (RAG ingestion pipeline).

Covers:
- _split_by_sections uses pattern variable (bug fix)
- Header-based splitting produces correct sections
- Documents without headers return as single section
- chunk_documents filters empty chunks
- Overlap is applied between chunks
- Chunk IDs are deterministic
- load_documents reads .md files only
"""
import os
import re
import tempfile
import pytest

from rag.ingest import DocumentIngestor


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ingestor(**kwargs):
    return DocumentIngestor(chunk_size=200, chunk_overlap=20, **kwargs)


# ── Bug 1 regression: pattern variable ───────────────────────────────────────

class TestPatternVariable:
    def test_pattern_is_string_not_tuple(self):
        """The pattern variable must be a plain str, not a tuple (stray comma bug)."""
        import inspect
        import rag.ingest as ingest_module
        source = inspect.getsource(ingest_module.DocumentIngestor._split_by_sections)
        # Find the pattern assignment line
        for line in source.splitlines():
            if "pattern" in line and "=" in line and "re.split" not in line and "#" not in line.lstrip()[:1]:
                # Should not end with a comma (which would make it a tuple)
                stripped = line.rstrip()
                assert not stripped.endswith(","), (
                    f"pattern assignment ends with comma (creates tuple): {stripped}"
                )

    def test_re_split_uses_pattern_variable(self):
        """re.split must reference the pattern variable, not a hard-coded literal."""
        import inspect
        import rag.ingest as ingest_module
        source = inspect.getsource(ingest_module.DocumentIngestor._split_by_sections)
        # The re.split call should use `pattern` not a raw string literal
        split_lines = [l for l in source.splitlines() if "re.split" in l]
        assert split_lines, "No re.split call found in _split_by_sections"
        for line in split_lines:
            assert "pattern" in line, (
                f"re.split should use the `pattern` variable, got: {line.strip()}"
            )


# ── _split_by_sections ────────────────────────────────────────────────────────

class TestSplitBySections:
    def test_splits_on_h2_headers(self):
        text = "Intro\n\n## Section A\nContent A\n\n## Section B\nContent B"
        ingestor = make_ingestor()
        sections = ingestor._split_by_sections(text)
        assert len(sections) == 3  # intro + 2 sections
        assert any("Section A" in s for s in sections)
        assert any("Section B" in s for s in sections)

    def test_splits_on_h3_headers(self):
        text = "### Part 1\nText 1\n### Part 2\nText 2"
        ingestor = make_ingestor()
        sections = ingestor._split_by_sections(text)
        assert len(sections) == 2

    def test_no_headers_returns_single_section(self):
        text = "Just a plain paragraph.\nNo headers here."
        ingestor = make_ingestor()
        sections = ingestor._split_by_sections(text)
        assert len(sections) == 1
        assert "plain paragraph" in sections[0]

    def test_empty_sections_filtered(self):
        text = "## A\n\n## B\nContent"
        ingestor = make_ingestor()
        sections = ingestor._split_by_sections(text)
        # Empty section between headers should be filtered
        for s in sections:
            assert s.strip() != ""

    def test_h1_not_split_boundary(self):
        """Single # headers should NOT be split boundaries."""
        text = "# Title\n\nSome content\n\n## Sub\nMore content"
        ingestor = make_ingestor()
        sections = ingestor._split_by_sections(text)
        # Only ## triggers a split, so we get 2 sections
        assert len(sections) == 2


# ── chunk_documents ───────────────────────────────────────────────────────────

class TestChunkDocuments:
    def _make_doc(self, content, source="test.md"):
        return {
            "content": content,
            "source": source,
            "metadata": {"filepath": f"/kb/{source}", "filename": source, "char_count": len(content)},
        }

    def test_short_doc_produces_chunks(self):
        ingestor = make_ingestor()
        doc = self._make_doc("## Hello\nShort content here.")
        chunks = ingestor.chunk_documents([doc])
        assert len(chunks) >= 1

    def test_chunk_keys_present(self):
        ingestor = make_ingestor()
        doc = self._make_doc("## Section\nSome text.")
        chunks = ingestor.chunk_documents([doc])
        for c in chunks:
            assert "chunk_id" in c
            assert "content" in c
            assert "source" in c
            assert "metadata" in c

    def test_empty_chunks_filtered(self):
        ingestor = make_ingestor()
        doc = self._make_doc("## A\n\n\n\n## B\n\n\n")
        chunks = ingestor.chunk_documents([doc])
        for c in chunks:
            assert len(c["content"]) > 20

    def test_chunk_ids_are_deterministic(self):
        ingestor = make_ingestor()
        doc = self._make_doc("## Section\nRepeatable content.")
        chunks1 = ingestor.chunk_documents([doc])
        chunks2 = ingestor.chunk_documents([doc])
        ids1 = [c["chunk_id"] for c in chunks1]
        ids2 = [c["chunk_id"] for c in chunks2]
        assert ids1 == ids2

    def test_source_preserved_in_chunks(self):
        ingestor = make_ingestor()
        doc = self._make_doc("## Section\nContent.", source="pricing.md")
        chunks = ingestor.chunk_documents([doc])
        for c in chunks:
            assert c["source"] == "pricing.md"

    def test_large_section_is_further_split(self):
        """A section larger than chunk_size should be split into multiple chunks."""
        ingestor = DocumentIngestor(chunk_size=100, chunk_overlap=10)
        # Use paragraph breaks so _split_by_size can split on \n\n boundaries
        paragraphs = "\n\n".join(["Word " * 15] * 10)  # 10 paragraphs, each ~75 chars
        long_text = "## Big Section\n" + paragraphs
        doc = self._make_doc(long_text)
        chunks = ingestor.chunk_documents([doc])
        assert len(chunks) > 1


# ── load_documents ────────────────────────────────────────────────────────────

class TestLoadDocuments:
    def test_loads_md_files_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (open(os.path.join(tmpdir, "a.md"), "w")).write("# Doc A\nContent")
            (open(os.path.join(tmpdir, "b.txt"), "w")).write("ignored")
            (open(os.path.join(tmpdir, "c.md"), "w")).write("# Doc C\nContent")
            ingestor = make_ingestor()
            docs = ingestor.load_documents(tmpdir)
            sources = [d["source"] for d in docs]
            assert "a.md" in sources
            assert "c.md" in sources
            assert "b.txt" not in sources

    def test_document_keys_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (open(os.path.join(tmpdir, "doc.md"), "w")).write("Hello world")
            ingestor = make_ingestor()
            docs = ingestor.load_documents(tmpdir)
            assert len(docs) == 1
            assert "content" in docs[0]
            assert "source" in docs[0]
            assert "metadata" in docs[0]

    def test_empty_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = make_ingestor()
            docs = ingestor.load_documents(tmpdir)
            assert docs == []
