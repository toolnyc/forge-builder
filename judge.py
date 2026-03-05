"""Forge Builder — Judge pipeline for evaluating aider outputs.

Collects free heuristic signals (exit code, git diff, ruff, output patterns)
and decides whether to ship, retry with a better prompt, or escalate to a
more capable model.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

log = logging.getLogger("forge-builder")

# ---------------------------------------------------------------------------
# Model chain — ordered cheapest → most expensive
# ---------------------------------------------------------------------------

MODEL_CHAIN: list[str] = [
    "ollama/qwen2.5-coder:7b",   # free, local
    "deepseek/deepseek-coder",    # ~$0.10/task
    "claude-sonnet-4-6",          # ~$1.00/task
]

MAX_RETRIES_PER_TIER = 2

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Verdict(Enum):
    PASS = "pass"
    FAIL_RETRY = "fail_retry"
    ESCALATE = "escalate"
    GIVE_UP = "give_up"


@dataclass
class AiderResult:
    exit_code: int
    stdout: str
    stderr: str
    cost_usd: float
    duration_s: float
    model: str


@dataclass
class JudgeSignals:
    exit_code: int = 0
    has_changes: bool = False
    has_src_changes: bool = False
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    ruff_errors: int = 0
    ruff_files: list[str] = field(default_factory=list)
    has_token_error: bool = False
    has_context_error: bool = False
    has_file_not_found: bool = False
    has_content_policy: bool = False
    has_todo_comments: bool = False
    todo_locations: list[str] = field(default_factory=list)


@dataclass
class JudgeResult:
    verdict: Verdict
    reason: str
    signals: JudgeSignals
    retry_hint: str | None = None


# ---------------------------------------------------------------------------
# Error pattern regexes (compiled once)
# ---------------------------------------------------------------------------

_TOKEN_ERROR_RE = re.compile(
    r"(token limit|context.length|maximum context|too many tokens|"
    r"max_tokens|context_length_exceeded)",
    re.IGNORECASE,
)
_FILE_NOT_FOUND_RE = re.compile(
    r"(file not found|no such file|filenotfounderror|can't find|cannot find)",
    re.IGNORECASE,
)
_CONTENT_POLICY_RE = re.compile(
    r"(content.?policy|content.?filter|safety|refused)",
    re.IGNORECASE,
)
_TODO_RE = re.compile(
    r"^\+.*\b(TODO|FIXME|HACK|blocked|unclear)\b",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Signal collectors (all free — no LLM calls)
# ---------------------------------------------------------------------------

def _collect_git_diff_signals(signals: JudgeSignals, repo_dir: str):
    """Parse git diff --stat to count changed files and lines."""
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir,
        )
        output = result.stdout.strip()
        if not output:
            return

        signals.has_changes = True
        # Last line is summary: " 3 files changed, 10 insertions(+), 2 deletions(-)"
        lines = output.splitlines()
        for line in lines[:-1]:
            # Each file line: " path/to/file.py | 5 ++-"
            parts = line.strip().split("|")
            if len(parts) >= 1:
                filepath = parts[0].strip()
                signals.files_changed += 1
                if filepath.endswith(".py") and not filepath.endswith("test_"):
                    signals.has_src_changes = True

        summary_match = re.search(
            r"(\d+) insertions?\(\+\)", lines[-1],
        )
        if summary_match:
            signals.lines_added = int(summary_match.group(1))

        del_match = re.search(r"(\d+) deletions?\(-\)", lines[-1])
        if del_match:
            signals.lines_removed = int(del_match.group(1))

    except Exception:
        log.debug("Failed to collect git diff signals", exc_info=True)


def _collect_ruff_signals(signals: JudgeSignals, repo_dir: str):
    """Run ruff check on changed .py files only."""
    try:
        # Get list of changed python files
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir,
        )
        py_files = [
            f for f in result.stdout.strip().splitlines()
            if f.endswith(".py") and Path(repo_dir, f).exists()
        ]
        if not py_files:
            return

        result = subprocess.run(
            ["ruff", "check", "--output-format=json", *py_files],
            capture_output=True, text=True, cwd=repo_dir,
        )
        if result.stdout.strip():
            try:
                errors = json.loads(result.stdout)
                signals.ruff_errors = len(errors)
                signals.ruff_files = list({e.get("filename", "") for e in errors})
            except json.JSONDecodeError:
                # Fall back to counting lines
                signals.ruff_errors = len(result.stdout.strip().splitlines())

    except FileNotFoundError:
        log.debug("ruff not installed, skipping lint check")
    except Exception:
        log.debug("Failed to collect ruff signals", exc_info=True)


def _collect_output_signals(signals: JudgeSignals, aider_result: AiderResult):
    """Scan aider stdout/stderr for known error patterns."""
    combined = (aider_result.stdout or "") + "\n" + (aider_result.stderr or "")

    signals.has_token_error = bool(_TOKEN_ERROR_RE.search(combined))
    signals.has_context_error = signals.has_token_error  # alias
    signals.has_file_not_found = bool(_FILE_NOT_FOUND_RE.search(combined))
    signals.has_content_policy = bool(_CONTENT_POLICY_RE.search(combined))


def _collect_todo_signals(signals: JudgeSignals, repo_dir: str):
    """Check if aider left TODO/blocked comments in the diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir,
        )
        matches = _TODO_RE.findall(result.stdout)
        if matches:
            signals.has_todo_comments = True
            signals.todo_locations = matches[:5]  # cap at 5
    except Exception:
        log.debug("Failed to collect TODO signals", exc_info=True)


def collect_signals(aider_result: AiderResult, repo_dir: str) -> JudgeSignals:
    """Run all signal collectors and return combined signals."""
    signals = JudgeSignals(exit_code=aider_result.exit_code)

    _collect_git_diff_signals(signals, repo_dir)
    _collect_ruff_signals(signals, repo_dir)
    _collect_output_signals(signals, aider_result)
    _collect_todo_signals(signals, repo_dir)

    return signals


# ---------------------------------------------------------------------------
# Verdict decision tree
# ---------------------------------------------------------------------------

def determine_verdict(
    signals: JudgeSignals,
    attempt: int,
    tier_index: int,
) -> JudgeResult:
    """Apply the decision tree to produce a verdict.

    Args:
        signals: Collected heuristic signals.
        attempt: 1-based attempt number within this tier.
        tier_index: 0-based index into MODEL_CHAIN.
    """
    is_last_tier = tier_index >= len(MODEL_CHAIN) - 1
    exit_ok = signals.exit_code == 0

    # --- Exit failed ---
    if not exit_ok:
        if signals.has_token_error or signals.has_context_error:
            if is_last_tier:
                return JudgeResult(
                    verdict=Verdict.GIVE_UP,
                    reason="Token/context error on highest tier",
                    signals=signals,
                )
            return JudgeResult(
                verdict=Verdict.ESCALATE,
                reason="Token/context limit — model too small",
                signals=signals,
            )

        if signals.has_file_not_found:
            if attempt < MAX_RETRIES_PER_TIER:
                return JudgeResult(
                    verdict=Verdict.FAIL_RETRY,
                    reason="File not found — prompt problem",
                    signals=signals,
                    retry_hint=(
                        "The previous attempt failed because it couldn't find the right files. "
                        "List the directory structure first, then make changes to the correct paths."
                    ),
                )
            if is_last_tier:
                return JudgeResult(
                    verdict=Verdict.GIVE_UP,
                    reason="File not found — retries exhausted on all tiers",
                    signals=signals,
                )
            return JudgeResult(
                verdict=Verdict.ESCALATE,
                reason="File not found — retries exhausted at this tier",
                signals=signals,
            )

        if signals.has_content_policy:
            return JudgeResult(
                verdict=Verdict.GIVE_UP,
                reason="Content policy rejection — cannot proceed",
                signals=signals,
            )

        # Generic failure
        if attempt < MAX_RETRIES_PER_TIER:
            return JudgeResult(
                verdict=Verdict.FAIL_RETRY,
                reason=f"Exit code {signals.exit_code} — retrying",
                signals=signals,
                retry_hint="The previous attempt failed. Try a different approach.",
            )
        if is_last_tier:
            return JudgeResult(
                verdict=Verdict.GIVE_UP,
                reason=f"Exit code {signals.exit_code} — all tiers exhausted",
                signals=signals,
            )
        return JudgeResult(
            verdict=Verdict.ESCALATE,
            reason=f"Exit code {signals.exit_code} — retries exhausted at this tier",
            signals=signals,
        )

    # --- Exit OK but no changes ---
    if not signals.has_changes:
        if attempt < MAX_RETRIES_PER_TIER:
            return JudgeResult(
                verdict=Verdict.FAIL_RETRY,
                reason="No changes produced",
                signals=signals,
                retry_hint=(
                    "The previous attempt ran but made no changes. "
                    "Be more specific about file paths and what code to modify."
                ),
            )
        if is_last_tier:
            return JudgeResult(
                verdict=Verdict.GIVE_UP,
                reason="No changes — all tiers exhausted",
                signals=signals,
            )
        return JudgeResult(
            verdict=Verdict.ESCALATE,
            reason="No changes — model can't figure it out",
            signals=signals,
        )

    # --- Exit OK, has changes, check quality ---
    if signals.ruff_errors > 3:
        if is_last_tier:
            # Still pass on highest tier — lint can be fixed in review
            if signals.has_src_changes:
                return JudgeResult(
                    verdict=Verdict.PASS,
                    reason=f"Passing with {signals.ruff_errors} lint errors (highest tier)",
                    signals=signals,
                )
            return JudgeResult(
                verdict=Verdict.GIVE_UP,
                reason=f"{signals.ruff_errors} lint errors, no src changes — all tiers exhausted",
                signals=signals,
            )
        return JudgeResult(
            verdict=Verdict.ESCALATE,
            reason=f"{signals.ruff_errors} ruff errors — model doesn't follow conventions",
            signals=signals,
        )

    if 0 < signals.ruff_errors <= 3:
        ruff_file_list = ", ".join(signals.ruff_files) if signals.ruff_files else "changed files"
        if attempt < MAX_RETRIES_PER_TIER:
            return JudgeResult(
                verdict=Verdict.FAIL_RETRY,
                reason=f"{signals.ruff_errors} ruff errors — fixable",
                signals=signals,
                retry_hint=(
                    f"The previous attempt has {signals.ruff_errors} lint errors in "
                    f"{ruff_file_list}. Fix all ruff lint errors before finishing."
                ),
            )
        # If retries exhausted at this tier but lint is minor, still pass
        if signals.has_src_changes:
            return JudgeResult(
                verdict=Verdict.PASS,
                reason=f"Passing with {signals.ruff_errors} minor lint errors",
                signals=signals,
            )

    # --- Clean pass ---
    if signals.has_src_changes:
        return JudgeResult(
            verdict=Verdict.PASS,
            reason="Clean pass — changes look good",
            signals=signals,
        )

    # Has changes but no src changes (only config, docs, etc.)
    return JudgeResult(
        verdict=Verdict.PASS,
        reason="Pass — non-source changes only",
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Optional LLM judge (free via ollama)
# ---------------------------------------------------------------------------

def llm_judge_diff(
    diff_stat: str,
    issue_title: str,
    ollama_model: str = "qwen2.5-coder:7b",
) -> bool | None:
    """Ask a local LLM if the diff actually implements the issue.

    Returns True if it looks good, False if suspicious, None on error.
    Only called when heuristics say PASS but diff looks suspicious.
    """
    prompt = f"""You are a code review judge. Given a git diff summary and an issue title,
answer ONLY "yes" or "no": does this diff implement the issue?

Issue: {issue_title}

Diff summary:
{diff_stat[:1000]}

Answer (yes/no):"""

    try:
        result = subprocess.run(
            ["ollama", "run", ollama_model, prompt],
            capture_output=True, text=True, timeout=30,
        )
        answer = result.stdout.strip().lower()
        if "yes" in answer:
            return True
        if "no" in answer:
            return False
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        log.debug("LLM judge unavailable or timed out")
        return None
    except Exception:
        log.debug("LLM judge failed", exc_info=True)
        return None


def should_run_llm_judge(signals: JudgeSignals) -> bool:
    """Heuristic: run LLM judge when diff looks suspicious despite PASS."""
    if not signals.has_changes:
        return False
    # Suspicious: more deleted than added, or only a few lines changed
    if signals.lines_removed > signals.lines_added * 2:
        return True
    if signals.lines_added <= 2 and signals.files_changed <= 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------

def chain_from(starting_model: str) -> list[str]:
    """Return the model chain starting at (or above) the given model.

    If the starting model isn't in the chain, returns the full chain.
    """
    try:
        idx = MODEL_CHAIN.index(starting_model)
        return MODEL_CHAIN[idx:]
    except ValueError:
        # Unknown model — check if it's sonnet/deepseek/ollama variant
        m = starting_model.lower()
        if any(k in m for k in ("ollama",)):
            return MODEL_CHAIN[0:]
        if any(k in m for k in ("deepseek",)):
            return MODEL_CHAIN[1:]
        # Default: start at sonnet tier
        return MODEL_CHAIN[2:]
