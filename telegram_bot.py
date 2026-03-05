#!/usr/bin/env python3
"""Forge Builder — Telegram bot for remote control.

Commands:
    /status  — paused/building/idle, current issue, budget remaining, model
    /budget [daily|issue] [amount] — view or set budget
    /model [model_string] — view or set model
    /pause / /resume — control builder loop
    /issues — list pending forge-build issues
    /approve <PR#> — squash-merge a PR via gh
    /logs — last 10 entries from budget.jsonl
    /add <title> — create a new GitHub issue with forge-build label

Setup:
    1. Message @BotFather on Telegram → /newbot → name it → copy the token
    2. Send any message to your new bot
    3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates → find your chat_id
    4. Add to .env:
         TELEGRAM_BOT_TOKEN=<token>
         TELEGRAM_CHAT_ID=<chat_id>
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

if TYPE_CHECKING:
    from forge_builder import BuilderConfig, BuilderState

log = logging.getLogger("forge-builder.telegram")

# Module-level refs set by start_bot()
_cfg: BuilderConfig | None = None
_state: BuilderState | None = None
_app: Application | None = None


def _authorized(update: Update) -> bool:
    """Only allow messages from the configured chat ID."""
    if not _cfg or not update.effective_chat:
        return False
    return str(update.effective_chat.id) == _cfg.telegram_chat_id


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return

    from forge_builder import budget_remaining, get_daily_spend

    paused = _state.get("paused")
    current = _state.get("current_issue")
    model = _state.get("default_model")
    remaining = budget_remaining()
    spent = get_daily_spend()
    daily = _state.get("daily_budget_usd")

    if paused:
        status_str = "PAUSED"
    elif current:
        status_str = f"BUILDING #{current['number']}: {current['title']}"
    else:
        status_str = "IDLE"

    msg = (
        f"Status: {status_str}\n"
        f"Model: {model}\n"
        f"Budget: ${remaining:.2f} remaining (${spent:.2f} / ${daily:.2f})\n"
    )
    await update.message.reply_text(msg)


async def cmd_budget(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return

    from forge_builder import budget_remaining, get_daily_spend

    args = ctx.args or []
    if len(args) == 2:
        kind, amount = args[0], args[1]
        try:
            val = float(amount)
        except ValueError:
            await update.message.reply_text("Invalid amount.")
            return
        if kind == "daily":
            _state.set("daily_budget_usd", val)
            await update.message.reply_text(f"Daily budget set to ${val:.2f}")
        elif kind == "issue":
            _cfg.per_issue_budget_usd = val
            await update.message.reply_text(f"Per-issue budget set to ${val:.2f}")
        else:
            await update.message.reply_text("Usage: /budget [daily|issue] [amount]")
        return

    remaining = budget_remaining()
    spent = get_daily_spend()
    daily = _state.get("daily_budget_usd")
    await update.message.reply_text(
        f"Daily: ${spent:.2f} / ${daily:.2f} (${remaining:.2f} left)\n"
        f"Per-issue cap: ${_cfg.per_issue_budget_usd:.2f}"
    )


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return

    args = ctx.args or []
    if args:
        new_model = args[0]
        _state.set("default_model", new_model)
        await update.message.reply_text(f"Model set to: {new_model}")
    else:
        await update.message.reply_text(f"Current model: {_state.get('default_model')}")


async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    _state.set("paused", True)
    await update.message.reply_text("Builder paused.")


async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    _state.set("paused", False)
    await update.message.reply_text("Builder resumed.")


async def cmd_issues(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return

    try:
        result = subprocess.run(
            [
                "gh", "issue", "list",
                "--repo", _cfg.github_repo,
                "--label", _cfg.label_pending,
                "--state", "open",
                "--json", "number,title",
                "--limit", "10",
            ],
            capture_output=True, text=True,
        )
        issues = json.loads(result.stdout)
        if not issues:
            await update.message.reply_text("No pending forge-build issues.")
            return
        lines = [f"#{i['number']}: {i['title']}" for i in issues]
        await update.message.reply_text("Pending issues:\n" + "\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_approve(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return

    args = ctx.args or []
    if not args:
        await update.message.reply_text("Usage: /approve <PR#>")
        return

    pr_num = args[0].lstrip("#")
    try:
        subprocess.run(
            [
                "gh", "pr", "merge", pr_num,
                "--repo", _cfg.github_repo,
                "--squash", "--delete-branch",
            ],
            capture_output=True, text=True, check=True,
        )
        await update.message.reply_text(f"PR #{pr_num} merged and branch deleted.")
    except subprocess.CalledProcessError as e:
        await update.message.reply_text(f"Merge failed: {e.stderr[:500]}")


async def cmd_logs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return

    log_path = Path(_cfg.budget_log)
    if not log_path.exists():
        await update.message.reply_text("No budget log yet.")
        return

    lines = log_path.read_text().splitlines()[-10:]
    entries = []
    for line in lines:
        try:
            e = json.loads(line)
            entries.append(
                f"#{e.get('issue', '?')} | {e.get('model', '?')} | "
                f"${e.get('cost_usd', 0):.2f} | {e.get('duration_s', 0):.0f}s"
            )
        except json.JSONDecodeError:
            continue

    if not entries:
        await update.message.reply_text("No log entries.")
        return

    await update.message.reply_text("Recent builds:\n" + "\n".join(entries))


async def cmd_add(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return

    args = ctx.args or []
    if not args:
        await update.message.reply_text("Usage: /add <issue title>")
        return

    title = " ".join(args)
    try:
        result = subprocess.run(
            [
                "gh", "issue", "create",
                "--repo", _cfg.github_repo,
                "--title", title,
                "--body", "Created via Forge Builder Telegram bot.",
                "--label", _cfg.label_pending,
            ],
            capture_output=True, text=True, check=True,
        )
        await update.message.reply_text(f"Issue created: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        await update.message.reply_text(f"Failed: {e.stderr[:500]}")


def _make_notify_fn(app: Application, chat_id: str):
    """Create a notification function that sends Telegram messages."""
    import asyncio

    def notify_fn(event: str, message: str):
        try:
            loop = app.bot._local.loop  # noqa: SLF001
        except AttributeError:
            loop = None

        async def _send():
            await app.bot.send_message(chat_id=chat_id, text=f"[{event}] {message}")

        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(_send(), loop)
        else:
            asyncio.run(_send())

    return notify_fn


def start_bot(config: BuilderConfig, builder_state: BuilderState):
    """Start the Telegram bot (blocks — runs asyncio event loop)."""
    global _cfg, _state, _app

    _cfg = config
    _state = builder_state

    from forge_builder import set_notify

    app = Application.builder().token(config.telegram_bot_token).build()
    _app = app

    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("budget", cmd_budget))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("issues", cmd_issues))
    app.add_handler(CommandHandler("approve", cmd_approve))
    app.add_handler(CommandHandler("logs", cmd_logs))
    app.add_handler(CommandHandler("add", cmd_add))

    # Wire up notifications after app is built
    set_notify(_make_notify_fn(app, config.telegram_chat_id))

    log.info("Telegram bot starting (chat_id=%s)...", config.telegram_chat_id)
    app.run_polling(allowed_updates=Update.ALL_TYPES)
