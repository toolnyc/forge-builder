#!/usr/bin/env python3
"""Forge Builder — autonomous build loop.

Polls GitHub Issues labeled 'forge-build', runs aider to implement each one,
and opens a PR for review. Optionally controlled via Telegram bot.

Designed to run as a systemd service on a Hetzner VPS.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from judge import (
    AiderResult,
    Verdict,
    chain_from,
    collect_signals,
    determine_verdict,
    llm_judge_diff,
    should_run_llm_judge,
    MAX_RETRIES_PER_TIER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("forge-builder")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BuilderConfig:
    repo_dir: str = os.getenv("FORGE_REPO_DIR", "/opt/forge")
    github_repo: str = os.getenv("FORGE_GITHUB_REPO", "")  # e.g. "petejames/forge"
    poll_interval: int = int(os.getenv("FORGE_POLL_INTERVAL", "300"))  # 5 min
    label_pending: str = "forge-build"
    label_in_progress: str = "building"
    label_done: str = "pr-ready"
    max_concurrent: int = 1
    branch_prefix: str = "forge-build"

    # --- Model config ---
    default_model: str = os.getenv("FORGE_DEFAULT_MODEL", "claude-sonnet-4-6")
    model_strategy: str = os.getenv("FORGE_MODEL_STRATEGY", "auto")
    aider_extra_args: str = os.getenv("FORGE_AIDER_EXTRA_ARGS", "")

    # --- Budget controls ---
    daily_budget_usd: float = float(os.getenv("FORGE_DAILY_BUDGET", "5.00"))
    per_issue_budget_usd: float = float(os.getenv("FORGE_PER_ISSUE_BUDGET", "1.50"))
    budget_log: str = os.getenv("FORGE_BUDGET_LOG", "/opt/forge/infra/builder/budget.jsonl")

    # --- Schedule controls ---
    active_hours: str = os.getenv("FORGE_ACTIVE_HOURS", "")

    # --- Telegram ---
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")


cfg = BuilderConfig()

# ---------------------------------------------------------------------------
# Shared state (thread-safe, mutated by Telegram bot + builder loop)
# ---------------------------------------------------------------------------

@dataclass
class BuilderState:
    paused: bool = False
    current_issue: dict | None = None
    daily_budget_usd: float = field(default_factory=lambda: cfg.daily_budget_usd)
    default_model: str = field(default_factory=lambda: cfg.default_model)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get(self, attr: str):
        with self._lock:
            return getattr(self, attr)

    def set(self, attr: str, value):
        with self._lock:
            setattr(self, attr, value)


state = BuilderState()

# Notification callback — set by telegram_bot when active
_notify_fn: Callable[[str, str], None] | None = None


def set_notify(fn: Callable[[str, str], None] | None):
    global _notify_fn
    _notify_fn = fn


def notify(event: str, message: str):
    """Send a notification (to Telegram if configured, otherwise just log)."""
    log.info("[%s] %s", event, message)
    if _notify_fn:
        try:
            _notify_fn(event, message)
        except Exception:
            log.exception("Notification failed")


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------

# Conservative max-cost ceiling per issue, used only for pre-flight checks.
# These are NOT used for logging — real costs come from aider output.
_MAX_COST_CEILING = {
    "cheap": 0.15,    # deepseek, groq
    "mid": 1.00,      # sonnet
    "expensive": 3.00, # opus
}


def estimate_max_cost(model: str) -> float:
    """Conservative ceiling for what an issue might cost with this model."""
    m = model.lower()
    if any(k in m for k in ("deepseek", "groq", "mixtral", "llama", "ollama")):
        return _MAX_COST_CEILING["cheap"]
    if "opus" in m:
        return _MAX_COST_CEILING["expensive"]
    return _MAX_COST_CEILING["mid"]


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_daily_spend() -> float:
    """Sum today's spend from the budget log."""
    log_path = Path(cfg.budget_log)
    if not log_path.exists():
        return 0.0

    today = _today_str()
    total = 0.0
    for line in log_path.read_text().splitlines():
        try:
            entry = json.loads(line)
            if entry.get("date") == today and entry.get("type", "spend") == "spend":
                total += entry.get("cost_usd", 0.0)
        except json.JSONDecodeError:
            continue
    return total


def log_spend(issue_number: int, model: str, cost_usd: float, duration_s: float):
    """Append a spend entry to the budget log."""
    log_path = Path(cfg.budget_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "type": "spend",
        "date": _today_str(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "issue": issue_number,
        "model": model,
        "cost_usd": round(cost_usd, 4),
        "duration_s": round(duration_s, 1),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_judgment(
    issue_number: int, attempt: int, model: str, verdict: str, reason: str,
):
    """Append a judgment entry to the budget log."""
    log_path = Path(cfg.budget_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "type": "judgment",
        "date": _today_str(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "issue": issue_number,
        "attempt": attempt,
        "model": model,
        "verdict": verdict,
        "reason": reason,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_issue_spend(issue_number: int) -> float:
    """Sum total spend for a specific issue from the budget log."""
    log_path = Path(cfg.budget_log)
    if not log_path.exists():
        return 0.0

    total = 0.0
    for line in log_path.read_text().splitlines():
        try:
            entry = json.loads(line)
            if entry.get("issue") == issue_number and entry.get("type", "spend") == "spend":
                total += entry.get("cost_usd", 0.0)
        except json.JSONDecodeError:
            continue
    return total


def budget_remaining() -> float:
    """How much budget is left today."""
    return state.get("daily_budget_usd") - get_daily_spend()


def is_within_active_hours() -> bool:
    """Check if current UTC hour is within configured active hours."""
    if not cfg.active_hours:
        return True

    try:
        start_str, end_str = cfg.active_hours.split("-")
        start_h, end_h = int(start_str), int(end_str)
    except ValueError:
        log.warning("Invalid FORGE_ACTIVE_HOURS format: %s (expected 'HH-HH')", cfg.active_hours)
        return True

    now_h = datetime.now(timezone.utc).hour
    if start_h <= end_h:
        return start_h <= now_h < end_h
    else:
        return now_h >= start_h or now_h < end_h


def can_afford_issue(issue: dict) -> tuple[bool, str]:
    """Pre-flight budget check. Returns (affordable, reason)."""
    model = pick_model(issue)
    ceiling = estimate_max_cost(model)
    remaining = budget_remaining()

    if remaining <= 0:
        return False, f"Daily budget exhausted (${state.get('daily_budget_usd'):.2f} spent)"

    if ceiling > remaining:
        return False, (
            f"Estimated max cost ${ceiling:.2f} ({model}) "
            f"exceeds remaining budget ${remaining:.2f}"
        )

    if ceiling > cfg.per_issue_budget_usd:
        # Will get downgraded, but still check the downgraded cost fits
        downgraded = _downgrade_model(model)
        downgraded_ceiling = estimate_max_cost(downgraded)
        if downgraded_ceiling > remaining:
            return False, (
                f"Even downgraded to {downgraded} (${downgraded_ceiling:.2f}), "
                f"exceeds remaining ${remaining:.2f}"
            )

    return True, "ok"


def pick_model(issue: dict) -> str:
    """Choose a model based on issue labels and budget.

    Returns an aider-compatible model string.
    """
    if cfg.model_strategy != "auto":
        return state.get("default_model")

    labels = {l["name"].lower() for l in issue.get("labels", [])}
    remaining = budget_remaining()
    daily = state.get("daily_budget_usd")
    remaining_pct = remaining / daily if daily > 0 else 0

    # Budget pressure: downgrade chain
    if remaining_pct < 0.30:
        log.info("Budget tight (%.1f%% remaining) — using deepseek", remaining_pct * 100)
        return "deepseek/deepseek-coder"

    # Label-based routing
    cheap_labels = {"simple", "docs", "documentation", "typo"}
    if labels & cheap_labels:
        log.info("Simple/docs task — using deepseek")
        return "deepseek/deepseek-coder"

    expensive_labels = {"complex", "architecture", "refactor", "feature"}
    if labels & expensive_labels:
        log.info("Complex task — using %s", state.get("default_model"))
        return state.get("default_model")

    return state.get("default_model")


def _downgrade_model(model: str) -> str:
    """Return a cheaper model alternative."""
    m = model.lower()
    if any(k in m for k in ("deepseek", "groq", "mixtral", "llama", "ollama")):
        return model  # already cheap
    if "opus" in m:
        return "claude-sonnet-4-6"
    # sonnet → deepseek
    return "deepseek/deepseek-coder"


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def run_cmd(
    cmd: list[str], cwd: str | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a subprocess, logging the command."""
    log.debug("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or cfg.repo_dir)
    if check and result.returncode != 0:
        log.error(
            "Command failed: %s\nstdout: %s\nstderr: %s",
            " ".join(cmd), result.stdout, result.stderr,
        )
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def gh(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a gh CLI command."""
    return run_cmd(["gh"] + args, **kwargs)

# ---------------------------------------------------------------------------
# GitHub Issue management
# ---------------------------------------------------------------------------

def fetch_pending_issues() -> list[dict]:
    """Get open issues labeled 'forge-build' that aren't already being worked on."""
    result = gh([
        "issue", "list",
        "--repo", cfg.github_repo,
        "--label", cfg.label_pending,
        "--state", "open",
        "--json", "number,title,body,labels",
        "--limit", "20",
    ])
    issues = json.loads(result.stdout)
    return [
        i for i in issues
        if not any(l["name"] == cfg.label_in_progress for l in i.get("labels", []))
    ]


def label_issue(number: int, add: list[str] | None = None, remove: list[str] | None = None):
    """Add/remove labels on an issue."""
    if add:
        for label in add:
            gh(["issue", "edit", str(number), "--repo", cfg.github_repo, "--add-label", label])
    if remove:
        for label in remove:
            gh(
                ["issue", "edit", str(number), "--repo", cfg.github_repo, "--remove-label", label],
                check=False,
            )


def comment_on_issue(number: int, body: str):
    """Post a comment on an issue."""
    gh(["issue", "comment", str(number), "--repo", cfg.github_repo, "--body", body])

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Turn text into a branch-safe slug."""
    slug = text.lower().strip()
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
    slug = "-".join(slug.split())
    return slug[:50]


def prepare_branch(issue_number: int, title: str) -> str:
    """Reset to main, pull, and create a feature branch."""
    branch_name = f"{cfg.branch_prefix}/{issue_number}-{slugify(title)}"

    run_cmd(["git", "fetch", "origin"])
    run_cmd(["git", "checkout", "main"], check=False)
    run_cmd(["git", "reset", "--hard", "origin/main"])
    run_cmd(["git", "branch", "-D", branch_name], check=False)
    run_cmd(["git", "checkout", "-b", branch_name])

    return branch_name


def has_changes() -> bool:
    """Check if there are uncommitted changes."""
    result = run_cmd(["git", "status", "--porcelain"], check=False)
    return bool(result.stdout.strip())


def commit_and_push(branch_name: str, issue_number: int, title: str):
    """Stage all changes, commit, push."""
    # Ensure secrets and logs are never committed
    run_cmd(
        ["git", "reset", "HEAD", "--", "infra/builder/.env", "infra/builder/budget.jsonl"],
        check=False,
    )
    run_cmd(["git", "checkout", "--", "infra/builder/.env"], check=False)
    run_cmd(["git", "add", "-A"])
    commit_msg = (
        f"forge-build: {title} (#{issue_number})\n\n"
        f"Autonomously implemented by Forge Builder using aider.\n"
        f"Closes #{issue_number}\n\n"
        f"Co-Authored-By: Forge Builder <forge-builder@noreply.github.com>"
    )
    run_cmd(["git", "commit", "-m", commit_msg])
    run_cmd(["git", "push", "-u", "origin", branch_name])


def open_pr(branch_name: str, issue_number: int, title: str, summary: str) -> str:
    """Open a PR and return its URL."""
    pr_body = f"""## Auto-generated by Forge Builder

Implements #{issue_number}

{summary}

---
Built autonomously by [Forge Builder](https://github.com/{cfg.github_repo}) using aider.
Review carefully before merging.
"""
    result = gh([
        "pr", "create",
        "--repo", cfg.github_repo,
        "--base", "main",
        "--head", branch_name,
        "--title", f"[forge-build] {title}",
        "--body", pr_body,
    ])
    return result.stdout.strip()

# ---------------------------------------------------------------------------
# Aider execution
# ---------------------------------------------------------------------------

_AIDER_COST_RE = re.compile(r"Cost:\s+\$[\d.]+\s+message,\s+\$([\d.]+)\s+session")


def _parse_aider_cost(output: str) -> float | None:
    """Extract session cost from aider stdout.

    Aider prints lines like: Cost: $0.12 message, $0.45 session
    We want the last session cost (cumulative).
    """
    matches = _AIDER_COST_RE.findall(output)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    return None


def run_aider(
    issue: dict,
    model: str,
    extra_context: str | None = None,
) -> AiderResult:
    """Run aider on the repo to implement the issue.

    Args:
        issue: GitHub issue dict with number, title, body.
        model: Aider-compatible model string.
        extra_context: Optional hint appended to prompt (from judge retries).

    Returns:
        AiderResult with exit code, output, cost, and timing.
    """
    prompt = f"""You are working on the Forge project — an open-source agent orchestrator.

Implement the following GitHub issue:

## #{issue['number']}: {issue['title']}

{issue.get('body', '') or 'No description provided.'}

---

Instructions:
- Read CLAUDE.md first for project conventions
- Make focused, minimal changes to implement this issue
- Write tests if the project has a test framework set up
- Do NOT modify unrelated code
- If you're blocked or something is unclear, leave a TODO comment explaining why
"""

    if extra_context:
        prompt += f"\n\nAdditional context from previous attempt:\n{extra_context}\n"

    cmd = [
        "aider",
        "--yes",
        "--no-auto-commits",
        "--model", model,
        "--message", prompt,
    ]

    # Add any extra args from config
    if cfg.aider_extra_args:
        cmd.extend(cfg.aider_extra_args.split())

    remaining = budget_remaining()
    log.info(
        "Running aider (model=%s, budget_remaining=$%.2f)...",
        model, remaining,
    )

    start_time = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cfg.repo_dir,
            timeout=1800,  # 30 min max per issue
        )
    except subprocess.TimeoutExpired:
        return AiderResult(
            exit_code=1,
            stdout="",
            stderr="Timed out after 30 minutes",
            cost_usd=0.0,
            duration_s=time.monotonic() - start_time,
            model=model,
        )
    duration_s = time.monotonic() - start_time

    # Parse real cost from aider output
    combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
    actual_cost = _parse_aider_cost(combined_output) or 0.0

    # Log real spend
    log_spend(issue["number"], model, actual_cost, duration_s)

    if result.returncode != 0:
        log.error(
            "Aider failed (model=%s):\nstdout: %s\nstderr: %s",
            model, result.stdout[-2000:], result.stderr[-2000:],
        )

    return AiderResult(
        exit_code=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
        cost_usd=actual_cost,
        duration_s=duration_s,
        model=model,
    )

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _reset_working_tree(branch_name: str):
    """Hard-reset the working tree for a retry attempt."""
    run_cmd(["git", "checkout", "--", "."], check=False)
    run_cmd(["git", "clean", "-fd"], check=False)
    run_cmd(["git", "reset", "--hard", f"origin/main"], check=False)
    run_cmd(["git", "checkout", branch_name], check=False)


def process_issue(issue: dict):
    """Process a single issue with judge-based retry/escalation.

    Flow:
    1. Pre-flight budget check
    2. Prepare branch
    3. For each model tier in the chain:
         For up to MAX_RETRIES_PER_TIER attempts:
           - Budget gate
           - Run aider
           - Collect signals → determine verdict
           - PASS → commit, PR, done
           - FAIL_RETRY → retry with hint
           - ESCALATE → next tier
    4. GIVE_UP → comment on issue, clean up
    """
    number = issue["number"]
    title = issue["title"]
    log.info("=== Processing issue #%d: %s ===", number, title)

    # --- Pre-flight budget check (BEFORE any labels/comments) ---
    affordable, reason = can_afford_issue(issue)
    if not affordable:
        log.info("Skipping issue #%d: %s", number, reason)
        return

    # Mark as in-progress (only after we know we can afford it)
    state.set("current_issue", issue)
    label_issue(number, add=[cfg.label_in_progress])
    starting_model = pick_model(issue)
    chain = chain_from(starting_model)
    comment_on_issue(
        number,
        f"Forge Builder is working on this... (starting with `{chain[0]}`)",
    )
    notify("build_start", f"Starting #{number}: {title} [{chain[0]}]")

    branch_name = None
    total_attempts = 0
    final_verdict = None

    try:
        branch_name = prepare_branch(number, title)

        extra_context: str | None = None

        for tier_index, model in enumerate(chain):
            for attempt in range(1, MAX_RETRIES_PER_TIER + 1):
                total_attempts += 1

                # Per-issue budget gate
                issue_spend = get_issue_spend(number)
                if issue_spend >= cfg.per_issue_budget_usd:
                    log.info(
                        "Issue #%d hit per-issue cap ($%.2f spent)",
                        number, issue_spend,
                    )
                    final_verdict = "give_up"
                    break

                # Daily budget gate
                if budget_remaining() <= 0:
                    log.info("Daily budget exhausted during issue #%d", number)
                    final_verdict = "give_up"
                    break

                # Reset working tree on retries (not first attempt)
                if total_attempts > 1:
                    _reset_working_tree(branch_name)

                log.info(
                    "Attempt %d (tier %d/%d, model=%s) for #%d",
                    total_attempts, tier_index + 1, len(chain), model, number,
                )

                aider_result = run_aider(issue, model, extra_context)
                signals = collect_signals(aider_result, cfg.repo_dir)
                judge_result = determine_verdict(signals, attempt, tier_index)

                # Optional LLM judge on suspicious PASSes
                if (
                    judge_result.verdict == Verdict.PASS
                    and should_run_llm_judge(signals)
                ):
                    ollama_model = os.getenv("FORGE_OLLAMA_MODEL", "qwen2.5-coder:7b")
                    diff_stat_result = run_cmd(
                        ["git", "diff", "--stat", "HEAD"], check=False,
                    )
                    llm_ok = llm_judge_diff(
                        diff_stat_result.stdout, title, ollama_model,
                    )
                    if llm_ok is False:
                        log.info("LLM judge rejected diff for #%d", number)
                        judge_result = judge_result.__class__(
                            verdict=Verdict.FAIL_RETRY
                            if attempt < MAX_RETRIES_PER_TIER
                            else Verdict.ESCALATE,
                            reason="LLM judge rejected: diff doesn't implement the issue",
                            signals=signals,
                            retry_hint=(
                                "The previous changes don't appear to actually implement "
                                "the issue. Re-read the issue carefully and make substantive "
                                "code changes."
                            ),
                        )

                log_judgment(
                    number, total_attempts, model,
                    judge_result.verdict.value, judge_result.reason,
                )
                log.info(
                    "Verdict for #%d attempt %d: %s — %s",
                    number, total_attempts,
                    judge_result.verdict.value, judge_result.reason,
                )

                if judge_result.verdict == Verdict.PASS:
                    # Ship it
                    summary = aider_result.stdout[-3000:] or "No output captured."
                    total_cost = get_issue_spend(number)

                    commit_and_push(branch_name, number, title)
                    pr_url = open_pr(branch_name, number, title, summary)

                    attempts_note = (
                        f" ({total_attempts} attempts, chain: "
                        f"{chain[0]} → {model})"
                        if total_attempts > 1
                        else ""
                    )
                    label_issue(
                        number,
                        add=[cfg.label_done],
                        remove=[cfg.label_in_progress, cfg.label_pending],
                    )
                    comment_on_issue(
                        number,
                        f"PR opened: {pr_url}{attempts_note}\n\n"
                        f"Please review before merging.",
                    )
                    log.info("PR created: %s", pr_url)
                    notify(
                        "build_success",
                        f"PR opened for #{number}: {title}\n{pr_url}\n"
                        f"Cost: ${total_cost:.2f}, attempts: {total_attempts}",
                    )
                    return

                if judge_result.verdict == Verdict.FAIL_RETRY:
                    extra_context = judge_result.retry_hint
                    continue

                if judge_result.verdict == Verdict.ESCALATE:
                    extra_context = (
                        f"Previous attempt with {model} failed: {judge_result.reason}. "
                        f"You are now running on a more capable model."
                    )
                    break  # break attempt loop, continue tier loop

                if judge_result.verdict == Verdict.GIVE_UP:
                    final_verdict = "give_up"
                    break

            # Check if we should stop entirely
            if final_verdict == "give_up":
                break
            # If inner loop exhausted retries without ESCALATE/GIVE_UP,
            # the last verdict handles it via the decision tree

        # --- All tiers exhausted or gave up ---
        total_cost = get_issue_spend(number)
        comment_on_issue(
            number,
            f"Forge Builder couldn't complete this issue after {total_attempts} attempts "
            f"across {len(chain)} model tiers.\n\n"
            f"Last verdict: {judge_result.reason if judge_result else 'unknown'}\n"
            f"Total cost: ${total_cost:.2f}\n\n"
            f"This may need a more specific description or manual implementation.",
        )
        label_issue(number, remove=[cfg.label_in_progress])
        run_cmd(["git", "checkout", "main"], check=False)
        if branch_name:
            run_cmd(["git", "branch", "-D", branch_name], check=False)
        notify(
            "build_fail",
            f"Gave up on #{number}: {title} "
            f"(${total_cost:.2f}, {total_attempts} attempts)",
        )

    except Exception as e:
        log.exception("Error processing issue #%d", number)
        comment_on_issue(number, f"Forge Builder encountered an error:\n```\n{e}\n```")
        label_issue(number, remove=[cfg.label_in_progress])
        run_cmd(["git", "checkout", "main"], check=False)
        notify("build_fail", f"Error on #{number}: {e}")
    finally:
        state.set("current_issue", None)


def post_run_summary():
    """Post a summary of today's build activity as a GitHub issue."""
    log_path = Path(cfg.budget_log)
    if not log_path.exists():
        return

    today = _today_str()
    spend_entries = []
    judgment_entries = []
    for line in log_path.read_text().splitlines():
        try:
            entry = json.loads(line)
            if entry.get("date") != today:
                continue
            if entry.get("type") == "judgment":
                judgment_entries.append(entry)
            else:
                spend_entries.append(entry)
        except json.JSONDecodeError:
            continue

    if not spend_entries and not judgment_entries:
        return

    entries = spend_entries  # backward compat for the rest of the summary
    total_cost = sum(e.get("cost_usd", 0) for e in spend_entries)
    issues_worked = list({e.get("issue") for e in spend_entries})
    models_used = {e.get("model", "unknown") for e in spend_entries}
    escalations = sum(1 for j in judgment_entries if j.get("verdict") == "escalate")

    summary = f"""## Forge Builder — Run Summary ({today})

| Metric | Value |
|--------|-------|
| Aider runs | {len(spend_entries)} |
| Issues worked | {', '.join(f'#{i}' for i in issues_worked)} |
| Models used | {', '.join(models_used)} |
| Escalations | {escalations} |
| Total cost | ${total_cost:.2f} |
| Budget remaining | ${budget_remaining():.2f} / ${state.get('daily_budget_usd'):.2f} |

### Activity Log
"""
    for e in spend_entries:
        dur = e.get("duration_s", 0)
        summary += (
            f"- **#{e.get('issue')}** — {e.get('model', '?')}, "
            f"${e.get('cost_usd', 0):.2f}, {dur:.0f}s\n"
        )

    if judgment_entries:
        summary += "\n### Judgments\n"
        for j in judgment_entries:
            summary += (
                f"- **#{j.get('issue')}** attempt {j.get('attempt', '?')} "
                f"({j.get('model', '?')}): **{j.get('verdict', '?')}** — "
                f"{j.get('reason', '')}\n"
            )

    summary += "\n---\n_Generated automatically by Forge Builder_"

    try:
        gh([
            "issue", "create",
            "--repo", cfg.github_repo,
            "--title", f"[build-summary] {today}",
            "--body", summary,
            "--label", "build-summary",
        ])
        log.info("Posted run summary for %s", today)
    except Exception:
        log.exception("Failed to post run summary")


def builder_loop():
    """Main polling loop. Runs in its own thread when Telegram is active."""
    log.info(
        "Forge Builder starting — repo: %s, poll: %ds, daily_budget: $%.2f, "
        "per_issue: $%.2f, hours: %s, model: %s",
        cfg.github_repo, cfg.poll_interval, cfg.daily_budget_usd,
        cfg.per_issue_budget_usd, cfg.active_hours or "always",
        cfg.default_model,
    )

    if not cfg.github_repo:
        log.error("FORGE_GITHUB_REPO not set. Exiting.")
        sys.exit(1)

    if not Path(cfg.repo_dir).exists():
        log.error("Repo dir %s does not exist. Exiting.", cfg.repo_dir)
        sys.exit(1)

    issues_processed = 0

    while True:
        try:
            # Pause gate (from Telegram /pause)
            if state.get("paused"):
                time.sleep(cfg.poll_interval)
                continue

            # Schedule gate
            if not is_within_active_hours():
                log.info("Outside active hours (%s). Sleeping...", cfg.active_hours)
                if issues_processed > 0:
                    post_run_summary()
                    issues_processed = 0
                time.sleep(cfg.poll_interval)
                continue

            # Budget gate
            remaining = budget_remaining()
            if remaining <= 0:
                log.info(
                    "Daily budget exhausted ($%.2f spent). Posting summary...",
                    state.get("daily_budget_usd"),
                )
                if issues_processed > 0:
                    post_run_summary()
                    issues_processed = 0
                notify("budget_hit", f"Daily budget exhausted. ${get_daily_spend():.2f} spent.")
                time.sleep(cfg.poll_interval * 6)
                continue

            log.info("Budget remaining today: $%.2f / $%.2f", remaining, state.get("daily_budget_usd"))

            issues = fetch_pending_issues()
            if not issues:
                log.info("No pending issues. Sleeping...")
                if issues_processed > 0:
                    post_run_summary()
                    issues_processed = 0
                time.sleep(cfg.poll_interval)
                continue

            log.info("Found %d pending issue(s)", len(issues))

            # Try issues in order — skip ones we can't afford
            any_affordable = False
            for issue in sorted(issues, key=lambda i: i["number"]):
                affordable, reason = can_afford_issue(issue)
                if affordable:
                    any_affordable = True
                    process_issue(issue)
                    issues_processed += 1
                    break
                else:
                    log.info("Can't afford #%d: %s", issue["number"], reason)

            if not any_affordable:
                log.info("No affordable issues in queue. Sleeping longer...")
                notify(
                    "budget_hit",
                    f"Can't afford any pending issues. "
                    f"Budget: ${budget_remaining():.2f} remaining.",
                )
                time.sleep(cfg.poll_interval * 6)
                continue

        except KeyboardInterrupt:
            log.info("Shutting down.")
            if issues_processed > 0:
                post_run_summary()
            break
        except Exception:
            log.exception("Unexpected error in main loop")

        time.sleep(cfg.poll_interval)


def main():
    """Entry point — start builder with optional Telegram bot."""
    if cfg.telegram_bot_token and cfg.telegram_chat_id:
        log.info("Telegram bot configured — starting bot + builder")
        # Import here to avoid hard dependency
        from telegram_bot import start_bot

        # Builder loop in daemon thread
        builder_thread = threading.Thread(target=builder_loop, daemon=True)
        builder_thread.start()

        # Telegram bot in main thread (asyncio event loop)
        start_bot(cfg, state)
    else:
        log.info("No Telegram config — running builder loop directly")
        builder_loop()


if __name__ == "__main__":
    main()
