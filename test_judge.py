"""Tests for the judge pipeline — verdict decision tree and signal collection."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from judge import (
    AiderResult,
    JudgeSignals,
    Verdict,
    chain_from,
    collect_signals,
    determine_verdict,
    should_run_llm_judge,
    MODEL_CHAIN,
    MAX_RETRIES_PER_TIER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    cost_usd: float = 0.0,
    model: str = "ollama/qwen2.5-coder:7b",
) -> AiderResult:
    return AiderResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        cost_usd=cost_usd,
        duration_s=10.0,
        model=model,
    )


def make_signals(**kwargs) -> JudgeSignals:
    return JudgeSignals(**kwargs)


# ---------------------------------------------------------------------------
# Decision tree: PASS cases
# ---------------------------------------------------------------------------

class TestVerdictPass:
    def test_clean_pass(self):
        signals = make_signals(
            exit_code=0, has_changes=True, has_src_changes=True,
            files_changed=3, lines_added=20, ruff_errors=0,
        )
        result = determine_verdict(signals, attempt=1, tier_index=0)
        assert result.verdict == Verdict.PASS

    def test_pass_non_source_changes(self):
        signals = make_signals(
            exit_code=0, has_changes=True, has_src_changes=False,
            files_changed=1, lines_added=5,
        )
        result = determine_verdict(signals, attempt=1, tier_index=0)
        assert result.verdict == Verdict.PASS

    def test_pass_with_minor_lint_after_retries_exhausted(self):
        signals = make_signals(
            exit_code=0, has_changes=True, has_src_changes=True,
            ruff_errors=2, ruff_files=["foo.py"],
        )
        result = determine_verdict(signals, attempt=MAX_RETRIES_PER_TIER, tier_index=0)
        assert result.verdict == Verdict.PASS

    def test_pass_with_many_lint_errors_on_highest_tier(self):
        signals = make_signals(
            exit_code=0, has_changes=True, has_src_changes=True,
            ruff_errors=10,
        )
        last_tier = len(MODEL_CHAIN) - 1
        result = determine_verdict(signals, attempt=1, tier_index=last_tier)
        assert result.verdict == Verdict.PASS


# ---------------------------------------------------------------------------
# Decision tree: FAIL_RETRY cases
# ---------------------------------------------------------------------------

class TestVerdictFailRetry:
    def test_minor_lint_errors_first_attempt(self):
        signals = make_signals(
            exit_code=0, has_changes=True, has_src_changes=True,
            ruff_errors=2, ruff_files=["a.py", "b.py"],
        )
        result = determine_verdict(signals, attempt=1, tier_index=0)
        assert result.verdict == Verdict.FAIL_RETRY
        assert "ruff" in result.reason.lower()
        assert result.retry_hint is not None

    def test_no_changes_first_attempt(self):
        signals = make_signals(exit_code=0, has_changes=False)
        result = determine_verdict(signals, attempt=1, tier_index=0)
        assert result.verdict == Verdict.FAIL_RETRY
        assert result.retry_hint is not None

    def test_file_not_found_first_attempt(self):
        signals = make_signals(exit_code=1, has_file_not_found=True)
        result = determine_verdict(signals, attempt=1, tier_index=0)
        assert result.verdict == Verdict.FAIL_RETRY
        assert "file" in result.reason.lower()

    def test_generic_failure_first_attempt(self):
        signals = make_signals(exit_code=1)
        result = determine_verdict(signals, attempt=1, tier_index=0)
        assert result.verdict == Verdict.FAIL_RETRY


# ---------------------------------------------------------------------------
# Decision tree: ESCALATE cases
# ---------------------------------------------------------------------------

class TestVerdictEscalate:
    def test_token_error_escalates(self):
        signals = make_signals(exit_code=1, has_token_error=True, has_context_error=True)
        result = determine_verdict(signals, attempt=1, tier_index=0)
        assert result.verdict == Verdict.ESCALATE

    def test_many_lint_errors_escalates(self):
        signals = make_signals(
            exit_code=0, has_changes=True, has_src_changes=True, ruff_errors=5,
        )
        result = determine_verdict(signals, attempt=1, tier_index=0)
        assert result.verdict == Verdict.ESCALATE

    def test_no_changes_second_attempt_escalates(self):
        signals = make_signals(exit_code=0, has_changes=False)
        result = determine_verdict(signals, attempt=MAX_RETRIES_PER_TIER, tier_index=0)
        assert result.verdict == Verdict.ESCALATE

    def test_file_not_found_retries_exhausted_escalates(self):
        signals = make_signals(exit_code=1, has_file_not_found=True)
        result = determine_verdict(signals, attempt=MAX_RETRIES_PER_TIER, tier_index=0)
        assert result.verdict == Verdict.ESCALATE

    def test_generic_failure_retries_exhausted_escalates(self):
        signals = make_signals(exit_code=1)
        result = determine_verdict(signals, attempt=MAX_RETRIES_PER_TIER, tier_index=0)
        assert result.verdict == Verdict.ESCALATE


# ---------------------------------------------------------------------------
# Decision tree: GIVE_UP cases
# ---------------------------------------------------------------------------

class TestVerdictGiveUp:
    def test_token_error_on_last_tier(self):
        signals = make_signals(exit_code=1, has_token_error=True, has_context_error=True)
        last_tier = len(MODEL_CHAIN) - 1
        result = determine_verdict(signals, attempt=1, tier_index=last_tier)
        assert result.verdict == Verdict.GIVE_UP

    def test_content_policy_always_gives_up(self):
        signals = make_signals(exit_code=1, has_content_policy=True)
        result = determine_verdict(signals, attempt=1, tier_index=0)
        assert result.verdict == Verdict.GIVE_UP

    def test_no_changes_last_tier(self):
        signals = make_signals(exit_code=0, has_changes=False)
        last_tier = len(MODEL_CHAIN) - 1
        result = determine_verdict(signals, attempt=MAX_RETRIES_PER_TIER, tier_index=last_tier)
        assert result.verdict == Verdict.GIVE_UP

    def test_generic_failure_last_tier_retries_exhausted(self):
        signals = make_signals(exit_code=1)
        last_tier = len(MODEL_CHAIN) - 1
        result = determine_verdict(
            signals, attempt=MAX_RETRIES_PER_TIER, tier_index=last_tier,
        )
        assert result.verdict == Verdict.GIVE_UP

    def test_many_lint_no_src_changes_last_tier(self):
        signals = make_signals(
            exit_code=0, has_changes=True, has_src_changes=False, ruff_errors=5,
        )
        last_tier = len(MODEL_CHAIN) - 1
        result = determine_verdict(signals, attempt=1, tier_index=last_tier)
        assert result.verdict == Verdict.GIVE_UP


# ---------------------------------------------------------------------------
# chain_from()
# ---------------------------------------------------------------------------

class TestChainFrom:
    def test_starts_at_ollama(self):
        chain = chain_from("ollama/qwen2.5-coder:7b")
        assert chain == MODEL_CHAIN

    def test_starts_at_deepseek(self):
        chain = chain_from("deepseek/deepseek-coder")
        assert chain == MODEL_CHAIN[1:]

    def test_starts_at_sonnet(self):
        chain = chain_from("claude-sonnet-4-6")
        assert chain == MODEL_CHAIN[2:]
        assert len(chain) == 1

    def test_unknown_model_defaults_to_sonnet_tier(self):
        chain = chain_from("gpt-4o")
        assert chain == MODEL_CHAIN[2:]

    def test_unknown_ollama_model_starts_at_bottom(self):
        chain = chain_from("ollama/codellama:13b")
        assert chain == MODEL_CHAIN

    def test_unknown_deepseek_variant(self):
        chain = chain_from("deepseek/deepseek-chat")
        assert chain == MODEL_CHAIN[1:]


# ---------------------------------------------------------------------------
# should_run_llm_judge()
# ---------------------------------------------------------------------------

class TestShouldRunLlmJudge:
    def test_no_changes_skips(self):
        assert not should_run_llm_judge(make_signals(has_changes=False))

    def test_more_deleted_than_added(self):
        signals = make_signals(
            has_changes=True, lines_added=5, lines_removed=20,
        )
        assert should_run_llm_judge(signals)

    def test_tiny_change(self):
        signals = make_signals(
            has_changes=True, lines_added=1, files_changed=1,
        )
        assert should_run_llm_judge(signals)

    def test_normal_change_skips(self):
        signals = make_signals(
            has_changes=True, lines_added=30, lines_removed=5, files_changed=3,
        )
        assert not should_run_llm_judge(signals)


# ---------------------------------------------------------------------------
# Signal collection (mocked subprocesses)
# ---------------------------------------------------------------------------

class TestCollectSignals:
    def _mock_run(self, responses: dict):
        """Return a side_effect function that maps commands to responses."""
        def side_effect(cmd, **kwargs):
            key = cmd[1] if cmd[0] == "git" else cmd[0]
            if key in responses:
                return responses[key]
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return side_effect

    @patch("judge.subprocess.run")
    def test_collects_exit_code(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        aider = make_result(exit_code=42)
        signals = collect_signals(aider, "/tmp/fake")
        assert signals.exit_code == 42

    @patch("judge.subprocess.run")
    def test_detects_token_error(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        aider = make_result(
            exit_code=1,
            stderr="Error: context_length_exceeded - max tokens is 8192",
        )
        signals = collect_signals(aider, "/tmp/fake")
        assert signals.has_token_error
        assert signals.has_context_error

    @patch("judge.subprocess.run")
    def test_detects_file_not_found(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        aider = make_result(stderr="FileNotFoundError: No such file or directory")
        signals = collect_signals(aider, "/tmp/fake")
        assert signals.has_file_not_found

    @patch("judge.subprocess.run")
    def test_detects_content_policy(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        aider = make_result(stderr="content policy violation: refused to generate")
        signals = collect_signals(aider, "/tmp/fake")
        assert signals.has_content_policy

    @patch("judge.subprocess.run")
    def test_detects_todo_in_diff(self, mock_run):
        diff_output = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def hello():
+    # TODO: blocked on API key
     pass
"""
        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["git", "diff"] and "--stat" not in cmd and "--name-only" not in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout=diff_output, stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        mock_run.side_effect = side_effect
        aider = make_result(exit_code=0)
        signals = collect_signals(aider, "/tmp/fake")
        assert signals.has_todo_comments


# ---------------------------------------------------------------------------
# Full scenario tests (decision tree walkthrough)
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_simple_issue_passes_on_ollama(self):
        """Happy path: ollama produces clean code on first try."""
        signals = make_signals(
            exit_code=0, has_changes=True, has_src_changes=True,
            files_changed=2, lines_added=15, ruff_errors=0,
        )
        result = determine_verdict(signals, attempt=1, tier_index=0)
        assert result.verdict == Verdict.PASS

    def test_lint_retry_then_pass(self):
        """First attempt has lint errors, retry fixes them."""
        # Attempt 1: lint errors
        signals1 = make_signals(
            exit_code=0, has_changes=True, has_src_changes=True,
            ruff_errors=2, ruff_files=["main.py"],
        )
        r1 = determine_verdict(signals1, attempt=1, tier_index=0)
        assert r1.verdict == Verdict.FAIL_RETRY
        assert r1.retry_hint

        # Attempt 2: clean
        signals2 = make_signals(
            exit_code=0, has_changes=True, has_src_changes=True,
            ruff_errors=0,
        )
        r2 = determine_verdict(signals2, attempt=2, tier_index=0)
        assert r2.verdict == Verdict.PASS

    def test_ollama_fails_deepseek_succeeds(self):
        """Ollama can't do it, escalate to deepseek which succeeds."""
        # Ollama attempt 1: no changes
        s1 = make_signals(exit_code=0, has_changes=False)
        r1 = determine_verdict(s1, attempt=1, tier_index=0)
        assert r1.verdict == Verdict.FAIL_RETRY

        # Ollama attempt 2: still no changes
        s2 = make_signals(exit_code=0, has_changes=False)
        r2 = determine_verdict(s2, attempt=2, tier_index=0)
        assert r2.verdict == Verdict.ESCALATE

        # Deepseek attempt 1: clean pass
        s3 = make_signals(
            exit_code=0, has_changes=True, has_src_changes=True,
            files_changed=3, lines_added=25,
        )
        r3 = determine_verdict(s3, attempt=1, tier_index=1)
        assert r3.verdict == Verdict.PASS

    def test_full_chain_exhaustion(self):
        """All tiers fail — should give up on last tier."""
        last = len(MODEL_CHAIN) - 1
        signals = make_signals(exit_code=1)
        result = determine_verdict(
            signals, attempt=MAX_RETRIES_PER_TIER, tier_index=last,
        )
        assert result.verdict == Verdict.GIVE_UP
