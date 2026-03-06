#!/usr/bin/env python3
"""One-shot test run: process a single issue from a target repo.

Usage:
    ANTHROPIC_API_KEY=sk-... python test_run.py

Processes issue #1 from toolnyc/clubstack as a test of the full pipeline.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

# Override config BEFORE importing forge_builder
os.environ.setdefault("FORGE_GITHUB_REPO", "toolnyc/clubstack")
os.environ.setdefault("FORGE_REPO_DIR", os.path.expanduser("~/Code/clubstack"))
os.environ.setdefault("FORGE_BUDGET_LOG", os.path.join(tempfile.gettempdir(), "forge-test-budget.jsonl"))
os.environ.setdefault("FORGE_DAILY_BUDGET", "5.00")
os.environ.setdefault("FORGE_PER_ISSUE_BUDGET", "3.00")
os.environ.setdefault("FORGE_DEFAULT_MODEL", "claude-sonnet-4-6")
os.environ.setdefault("FORGE_MODEL_STRATEGY", "auto")

# Load API key from ra-killer .env if not already set
if not os.environ.get("ANTHROPIC_API_KEY"):
    env_path = os.path.expanduser("~/Code/ra-killer/.env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.strip().split("=", 1)[1]
                break

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set")
    sys.exit(1)

# Now import (config reads env at import time)
import forge_builder as fb
from forge_builder import cfg, state, log

def main():
    issue_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    log.info("=== FORGE BUILDER TEST RUN ===")
    log.info("Repo: %s", cfg.github_repo)
    log.info("Repo dir: %s", cfg.repo_dir)
    log.info("Model: %s", cfg.default_model)
    log.info("Budget log: %s", cfg.budget_log)
    log.info("Target issue: #%d", issue_number)

    # Fetch the specific issue
    result = fb.gh([
        "issue", "view", str(issue_number),
        "--repo", cfg.github_repo,
        "--json", "number,title,body,labels",
    ])
    issue = json.loads(result.stdout)
    log.info("Issue #%d: %s", issue["number"], issue["title"])

    # Quick single-attempt test: run aider directly and show output
    if "--debug" in sys.argv:
        branch = fb.prepare_branch(issue_number, issue["title"])
        result = fb.run_aider(issue, cfg.default_model)
        log.info("=== Aider stdout (last 3000 chars) ===")
        log.info(result.stdout[-3000:])
        log.info("=== Aider stderr (last 2000 chars) ===")
        log.info(result.stderr[-2000:])
        # Show what changed
        import subprocess
        diff = subprocess.run(
            ["git", "diff", "--stat", "origin/main"],
            capture_output=True, text=True, cwd=cfg.repo_dir,
        )
        log.info("=== Diff vs origin/main ===")
        log.info(diff.stdout)
        git_log = subprocess.run(
            ["git", "log", "--oneline", "origin/main..HEAD"],
            capture_output=True, text=True, cwd=cfg.repo_dir,
        )
        log.info("=== Commits on branch ===")
        log.info(git_log.stdout)
        return

    # Process it
    fb.process_issue(issue)

    # Show budget log
    from pathlib import Path
    log_path = Path(cfg.budget_log)
    if log_path.exists():
        log.info("=== Budget log ===")
        for line in log_path.read_text().splitlines():
            log.info(line)

    log.info("=== TEST RUN COMPLETE ===")


if __name__ == "__main__":
    main()
