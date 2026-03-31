"""
Tests for src/eval_dataset_labelling/llm_relevance_judge.py

Covers:
- truncate_chunk(): within and exceeding limit
- parse_llm_response(): valid JSON, markdown fences, fallback, invalid
- get_client(): missing API key raises
- judge_pair(): successful call, retry on transient error, parse failure

All OpenAI and GCS interactions are mocked.

Skipped if pandas or openai is not installed.
"""
import json
import pytest

pytest.importorskip("pandas", reason="pandas not installed")
pytest.importorskip("openai", reason="openai not installed")

from unittest.mock import MagicMock, patch

from src.eval_dataset_labelling.llm_relevance_judge import (
    truncate_chunk,
    parse_llm_response,
    get_client,
    judge_pair,
    LLM_PARAMS,
)


# ─────────────────────────────────────────────────
# truncate_chunk()
# ─────────────────────────────────────────────────

class TestTruncateChunk:
    def test_within_limit(self):
        text = "short text"
        assert truncate_chunk(text) == text

    def test_exceeds_limit(self):
        limit = LLM_PARAMS["max_chunk_chars"]
        text = "A" * (limit + 100)
        result = truncate_chunk(text)
        assert len(result) == limit + len("...")
        assert result.endswith("...")

    def test_exactly_at_limit(self):
        limit = LLM_PARAMS["max_chunk_chars"]
        text = "B" * limit
        assert truncate_chunk(text) == text


# ─────────────────────────────────────────────────
# parse_llm_response()
# ─────────────────────────────────────────────────

class TestParseLlmResponse:
    def test_valid_json(self):
        resp = json.dumps({"relevance": 2, "reason": "directly relevant"})
        relevance, reason = parse_llm_response(resp)
        assert relevance == 2
        assert reason == "directly relevant"

    def test_json_with_markdown_fences(self):
        resp = '```json\n{"relevance": 1, "reason": "partial"}\n```'
        relevance, reason = parse_llm_response(resp)
        assert relevance == 1
        assert reason == "partial"

    def test_bare_digit_fallback(self):
        resp = "The relevance is 0 because it is off topic."
        relevance, reason = parse_llm_response(resp)
        assert relevance == 0
        assert reason == "parse_fallback"

    def test_invalid_relevance_value_with_fallback(self):
        resp = json.dumps({"relevance": 5, "reason": "bad"})
        # 5 is invalid, falls through to bare digit search, finds "5" which is not 0/1/2
        with pytest.raises(ValueError, match="Could not parse"):
            parse_llm_response(resp)

    def test_completely_invalid_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_llm_response("no digits or json here")

    def test_relevance_zero(self):
        resp = json.dumps({"relevance": 0, "reason": "not relevant"})
        relevance, _ = parse_llm_response(resp)
        assert relevance == 0


# ─────────────────────────────────────────────────
# get_client()
# ─────────────────────────────────────────────────

class TestGetClient:
    def test_missing_api_key_raises(self):
        with patch.dict("src.eval_dataset_labelling.llm_relevance_judge.LLM_PARAMS", {"api_key": ""}):
            with pytest.raises(ValueError, match="API key"):
                get_client()

    @patch("src.eval_dataset_labelling.llm_relevance_judge.OpenAI")
    def test_valid_key_returns_client(self, mock_openai):
        with patch.dict("src.eval_dataset_labelling.llm_relevance_judge.LLM_PARAMS", {"api_key": "sk-test"}):
            client = get_client()
            mock_openai.assert_called_once_with(api_key="sk-test")


# ─────────────────────────────────────────────────
# judge_pair()
# ─────────────────────────────────────────────────

class TestJudgePair:
    def test_successful_call(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"relevance": 2, "reason": "direct match"}
        )
        mock_client.chat.completions.create.return_value = mock_response
        logger = MagicMock()

        relevance, reason = judge_pair(mock_client, "query", "chunk", logger)
        assert relevance == 2
        assert reason == "direct match"

    def test_all_retries_exhausted_returns_negative(self):
        from openai import RateLimitError

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="rate limited",
            response=mock_response,
            body=None,
        )
        logger = MagicMock()

        with patch.dict(
            "src.eval_dataset_labelling.llm_relevance_judge.LLM_PARAMS",
            {"max_retries": 2, "retry_delay": 0, **LLM_PARAMS},
        ):
            with patch("src.eval_dataset_labelling.llm_relevance_judge.time.sleep"):
                relevance, reason = judge_pair(mock_client, "q", "c", logger)
        assert relevance == -1
        assert reason == "max_retries_exceeded"

    def test_parse_failure_returns_negative(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "completely unparseable garbage no digits"
        mock_client.chat.completions.create.return_value = mock_response
        logger = MagicMock()

        with patch.dict(
            "src.eval_dataset_labelling.llm_relevance_judge.LLM_PARAMS",
            {"max_retries": 1, **LLM_PARAMS},
        ):
            relevance, reason = judge_pair(mock_client, "q", "c", logger)
        assert relevance == -1
        assert reason == "parse_failed"
