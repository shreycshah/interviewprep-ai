"""
Tests for src/chunking/chunker.py

Covers:
- wc(): word counting
- split_sentences(): sentence boundary splitting
- char_offset(): character offset lookup
- build_header(): context prefix construction
- get_round_label(): round header extraction from regex match
- detect_strategy(): strategy selection based on content
- fixed_window_split(): sliding window with overlap
- structural_split(): round-boundary splitting
- chunk_document(): end-to-end chunking
- validate_chunks(): chunk validation and error detection
"""
import re
import pytest

from unittest.mock import MagicMock

from src.chunking.chunker import (
    wc,
    split_sentences,
    char_offset,
    build_header,
    get_round_label,
    detect_strategy,
    fixed_window_split,
    structural_split,
    chunk_document,
    validate_chunks,
    CHUNK_SIZE_WORDS,
    OVERLAP_WORDS,
    MIN_DOC_WORDS,
    ROUND_RE,
)
from src.data_models.document_chunk import DocumentChunk


# ─────────────────────────────────────────────────
# wc()
# ─────────────────────────────────────────────────

class TestWordCount:
    def test_empty_string(self):
        assert wc("") == 0

    def test_single_word(self):
        assert wc("hello") == 1

    def test_multiple_words(self):
        assert wc("one two three four") == 4

    def test_extra_whitespace(self):
        assert wc("  hello   world  ") == 2


# ─────────────────────────────────────────────────
# split_sentences()
# ─────────────────────────────────────────────────

class TestSplitSentences:
    def test_single_sentence(self):
        result = split_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_sentences(self):
        result = split_sentences("First sentence. Second sentence. Third one!")
        assert len(result) == 3

    def test_empty_string(self):
        result = split_sentences("")
        assert result == []

    def test_preserves_exclamation_and_question(self):
        result = split_sentences("What? Yes! Done.")
        assert len(result) == 3

    def test_strips_leading_trailing_whitespace(self):
        result = split_sentences("  Hello world.  ")
        assert result[0] == "Hello world."


# ─────────────────────────────────────────────────
# char_offset()
# ─────────────────────────────────────────────────

class TestCharOffset:
    def test_found_segment(self):
        full = "Hello world, this is a test."
        segment = "this is a test."
        start, end = char_offset(full, segment)
        assert full[start:end] == segment

    def test_not_found_returns_zero_to_len(self):
        start, end = char_offset("Hello", "missing")
        assert start == 0
        assert end == len("missing")

    def test_segment_at_start(self):
        full = "Hello world"
        start, end = char_offset(full, "Hello")
        assert start == 0
        assert end == 5


# ─────────────────────────────────────────────────
# build_header()
# ─────────────────────────────────────────────────

class TestBuildHeader:
    def test_all_fields(self):
        header = build_header("Google", "SDE", "Round 1", 0, 3)
        assert "Company: Google" in header
        assert "Role: SDE" in header
        assert "Round: Round 1" in header
        assert "Part: 1 of 3" in header

    def test_no_optional_fields(self):
        header = build_header(None, None, None, 0, 1)
        assert "Part: 1 of 1" in header
        assert "Company" not in header
        assert "Role" not in header
        assert "Round" not in header

    def test_partial_fields(self):
        header = build_header("Amazon", None, None, 2, 5)
        assert "Company: Amazon" in header
        assert "Part: 3 of 5" in header
        assert "Role" not in header

    def test_ends_with_pipe_space(self):
        header = build_header(None, None, None, 0, 1)
        assert header.endswith("| ")


# ─────────────────────────────────────────────────
# get_round_label()
# ─────────────────────────────────────────────────

class TestGetRoundLabel:
    def test_extracts_round_label_from_text(self):
        text = "Some preamble.\nRound 1: Technical Interview\nSome content."
        match = re.search(r"\nRound 1", text)
        label = get_round_label(text, match)
        assert "Round 1" in label

    def test_truncates_long_labels(self):
        long_line = "Round 1: " + "A" * 100
        text = f"\n{long_line}\nNext line."
        match = re.search(r"\nRound 1", text)
        label = get_round_label(text, match)
        assert len(label) <= 60


# ─────────────────────────────────────────────────
# detect_strategy()
# ─────────────────────────────────────────────────

class TestDetectStrategy:
    def test_short_doc_returns_single_chunk(self):
        short = " ".join(["word"] * (MIN_DOC_WORDS - 1))
        assert detect_strategy(short) == "single_chunk"

    def test_long_doc_without_markers_returns_fixed_window(self):
        long = " ".join(["word"] * (MIN_DOC_WORDS + 100))
        assert detect_strategy(long) == "fixed_window"

    def test_doc_with_round_markers_returns_structural(self):
        text = " ".join(["word"] * (MIN_DOC_WORDS + 10))
        text += "\nRound 1: Technical\nSome content here."
        assert detect_strategy(text) == "structural"


# ─────────────────────────────────────────────────
# fixed_window_split()
# ─────────────────────────────────────────────────

class TestFixedWindowSplit:
    def test_short_text_returns_single_window(self):
        text = "This is a short sentence."
        windows = fixed_window_split(text)
        assert len(windows) == 1

    def test_long_text_produces_multiple_windows(self):
        sentences = [f"Sentence number {i}." for i in range(200)]
        text = " ".join(sentences)
        windows = fixed_window_split(text)
        assert len(windows) > 1

    def test_windows_have_overlap(self):
        sentences = [f"Sentence number {i} with extra padding words." for i in range(200)]
        text = " ".join(sentences)
        windows = fixed_window_split(text)
        if len(windows) >= 2:
            last_words_first = set(windows[0].split()[-OVERLAP_WORDS:])
            first_words_second = set(windows[1].split()[:OVERLAP_WORDS])
            assert len(last_words_first & first_words_second) > 0

    def test_empty_string(self):
        windows = fixed_window_split("")
        assert windows == []


# ─────────────────────────────────────────────────
# structural_split()
# ─────────────────────────────────────────────────

class TestStructuralSplit:
    def test_no_markers_returns_single_segment(self):
        text = "Just some text without any round markers."
        segments = structural_split(text)
        assert len(segments) == 1
        assert segments[0][0] is None

    def test_with_round_markers_splits_correctly(self):
        text = (
            "Some intro text.\n"
            "Round 1: Technical\n"
            "Round 1 content here.\n"
            "Round 2: System Design\n"
            "Round 2 content here."
        )
        segments = structural_split(text)
        assert len(segments) >= 2
        labels = [label for label, _ in segments if label and "Round" in label]
        assert len(labels) >= 2

    def test_preamble_captured(self):
        text = (
            "I interviewed at Google last month.\n"
            "Round 1: Phone screen\n"
            "They asked DSA questions."
        )
        segments = structural_split(text)
        labels = [label for label, _ in segments]
        assert "Preamble" in labels


# ─────────────────────────────────────────────────
# chunk_document()
# ─────────────────────────────────────────────────

class TestChunkDocument:
    def test_short_doc_single_chunk(self):
        text = "Short interview experience at Google."
        chunks = chunk_document("doc_1", text, "leetcode", company="Google")
        assert len(chunks) == 1
        assert chunks[0].strategy == "single_chunk"
        assert chunks[0].document_id == "doc_1"

    def test_chunk_has_header(self):
        text = "Short interview experience."
        chunks = chunk_document("doc_1", text, "leetcode", company="Google", role="SDE")
        assert "Company: Google" in chunks[0].chunk_text
        assert "Role: SDE" in chunks[0].chunk_text

    def test_long_doc_fixed_window(self):
        sentences = [f"Interview question {i} about algorithms and data structures." for i in range(200)]
        text = " ".join(sentences)
        chunks = chunk_document("doc_2", text, "gfg")
        assert len(chunks) > 1
        assert all(c.strategy == "fixed_window" for c in chunks)

    def test_structural_chunking(self):
        text = " ".join(["word"] * (MIN_DOC_WORDS + 10))
        text += "\nRound 1: Technical\nAsked about graphs.\nRound 2: Design\nSystem design question."
        chunks = chunk_document("doc_3", text, "medium")
        assert any(c.strategy == "structural" for c in chunks)

    def test_chunk_indices_sequential(self):
        sentences = [f"Sentence {i} about interview preparation." for i in range(200)]
        text = " ".join(sentences)
        chunks = chunk_document("doc_4", text, "leetcode")
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.total_chunks == len(chunks)

    def test_empty_topics_default(self):
        chunks = chunk_document("doc_5", "Short text.", "leetcode", topics=None)
        assert len(chunks) >= 1

    def test_chunk_id_is_uuid(self):
        chunks = chunk_document("doc_6", "Short text.", "leetcode")
        import uuid
        uuid.UUID(chunks[0].chunk_id)  # raises if invalid


# ─────────────────────────────────────────────────
# validate_chunks()
# ─────────────────────────────────────────────────

class TestValidateChunks:
    def _make_mock_chunk(self, **overrides):
        defaults = {
            "chunk_id": "c1",
            "document_id": "d1",
            "chunk_index": 0,
            "total_chunks": 1,
            "chunk_text": "Company: Google | Part: 1 of 1 | Some text.",
            "raw_text": "Some text.",
            "word_count": 2,
            "char_start_offset": 0,
            "char_end_offset": 10,
            "strategy": "single_chunk",
            "round_label": None,
            "company": "Google",
        }
        defaults.update(overrides)
        m = MagicMock()
        for k, v in defaults.items():
            setattr(m, k, v)
        return m

    def test_valid_chunks_pass(self):
        chunks = [self._make_mock_chunk()]
        report = validate_chunks(chunks)
        assert report["total"] == 1
        assert report["errors"] == []

    def test_missing_document_id_raises(self):
        chunks = [self._make_mock_chunk(document_id="")]
        with pytest.raises(AssertionError, match="Validation failed"):
            validate_chunks(chunks)

    def test_oversized_chunk_adds_warning(self):
        chunks = [self._make_mock_chunk(
            word_count=CHUNK_SIZE_WORDS + OVERLAP_WORDS + 10,
            company=None,
        )]
        report = validate_chunks(chunks)
        assert len(report["warnings"]) > 0

    def test_empty_list_passes(self):
        report = validate_chunks([])
        assert report["total"] == 0
        assert report["errors"] == []
