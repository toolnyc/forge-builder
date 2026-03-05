# Forge Builder — Autonomous CI Agent

Pete's autonomous build agent. Polls GitHub Issues labeled `forge-build`, runs aider (Claude Code headless) to implement each one, and opens PRs for review. Works on any repo, not just forge.

Runs on a Hetzner VPS as a systemd service. Optionally controlled via Telegram bot.

## How it works

1. Poll GitHub Issues labeled `forge-build`
2. Pick up next issue, create a feature branch
3. Run aider with the issue as prompt
4. Judge pipeline evaluates the output (heuristics + optional local LLM)
5. Pass → commit, push, open PR, label issue `pr-ready`
6. Fail → retry with better prompt, escalate to stronger model, or give up

## Files

```
forge-builder/
├── forge_builder.py        # Main loop: poll → build → PR
├── judge.py                # Judge pipeline (signals, verdict tree, LLM judge)
├── test_judge.py           # Tests for judge pipeline
├── telegram_bot.py         # Telegram bot for remote control
├── forge-builder.service   # systemd unit file
├── setup.sh                # Hetzner VPS provisioning script
├── requirements.txt        # Python dependencies
├── .env.example            # Config template
├── GOTCHAS.md              # VPS setup gotchas and rescue procedures
└── CLAUDE.md               # This file
```

## Budget controls

Budget is real — don't burn tokens carelessly.

- **Daily cap**: $5 default (`FORGE_DAILY_BUDGET`)
- **Per-issue cap**: $1.50 default (`FORGE_PER_ISSUE_BUDGET`)
- **Model auto-switching**: Issue labels drive model selection
  - `simple`/`docs` → deepseek (cheap)
  - `complex`/`feature` → sonnet (default)
  - Budget pressure (<30% remaining) → auto-downgrade to deepseek
- **Active hours**: Optional UTC window (`FORGE_ACTIVE_HOURS=22-06`)
- **Spend log**: Append-only `budget.jsonl`

## Judge pipeline (judge.py)

Model chain (cheapest → most expensive): `ollama/qwen2.5-coder:7b` → `deepseek/deepseek-coder` → `claude-sonnet-4-6`

Per tier, up to 2 retries. Verdicts:
- **PASS** → ship it
- **FAIL_RETRY** → retry with hint (lint errors, no changes, file not found)
- **ESCALATE** → next model tier (token limits, persistent failures)
- **GIVE_UP** → comment on issue, clean up

Signals collected (all free, no LLM calls): exit code, git diff stats, ruff lint, error patterns (token limit, file not found, content policy), TODO comments in diff.

Optional LLM judge via local Ollama on suspicious PASSes (more deleted than added, tiny changes).

## Telegram commands

`/status` — paused/building/idle, current issue, budget remaining
`/budget [daily|issue] [amount]` — view or set budget caps
`/model [model_string]` — view or set default model
`/pause` / `/resume` — control builder loop
`/issues` — list pending forge-build issues
`/approve <PR#>` — squash-merge a PR
`/logs` — last 10 budget.jsonl entries
`/add <title>` — create new issue with forge-build label

## Deployment

Runs on Hetzner VPS as systemd service. See GOTCHAS.md for actual VPS details vs what setup.sh assumes.

Auto-deploys on push to main via GitHub Actions (`deploy-builder.yml`). Requires secrets: `HETZNER_HOST`, `HETZNER_USER`, `HETZNER_SSH_KEY`.

## Conventions

- `from __future__ import annotations` in all Python files
- Type hints everywhere: `X | None` not `Optional[X]`
- snake_case for everything except class names
- Stay tightly scoped — don't expand beyond what was asked
