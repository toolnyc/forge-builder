#!/usr/bin/env bash
set -euo pipefail

# Forge Builder — Hetzner VPS setup script
# Run as root on a fresh Ubuntu 22.04+ box

echo "=== Forge Builder Setup ==="

# --- System user ---
if ! id -u forge &>/dev/null; then
    useradd -r -m -s /bin/bash forge
    echo "Created 'forge' system user"
fi

# --- System deps ---
apt-get update -qq
apt-get install -y -qq git python3 python3-pip curl jq

# --- GitHub CLI ---
if ! command -v gh &>/dev/null; then
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    apt-get update -qq && apt-get install -y -qq gh
    echo "Installed gh CLI"
fi

# --- Ollama (local LLM for free judge + cheap builds) ---
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Installed Ollama"
fi
# Pull the default model in the background (non-blocking)
sudo -u forge ollama pull qwen2.5-coder:7b &
echo "Pulling qwen2.5-coder:7b in background..."

# --- Aider + Python deps ---
pip3 install -r /opt/forge/infra/builder/requirements.txt
echo "Installed aider and Python dependencies"

# --- Clone repo ---
REPO_DIR="/opt/forge"
if [ ! -d "$REPO_DIR" ]; then
    echo "Enter your GitHub repo (e.g. youruser/forge):"
    read -r GITHUB_REPO
    git clone "https://github.com/$GITHUB_REPO.git" "$REPO_DIR"
    chown -R forge:forge "$REPO_DIR"
else
    echo "Repo already exists at $REPO_DIR"
fi

# --- Environment file ---
ENV_FILE="$REPO_DIR/infra/builder/.env"
if [ ! -f "$ENV_FILE" ]; then
    cp "$REPO_DIR/infra/builder/.env.example" "$ENV_FILE"
    echo ""
    echo "IMPORTANT: Edit $ENV_FILE with your values:"
    echo "  - ANTHROPIC_API_KEY"
    echo "  - FORGE_GITHUB_REPO"
    echo "  - Budget settings"
    echo ""
fi

# --- Git config for the forge user ---
sudo -u forge git config --global user.name "Forge Builder"
sudo -u forge git config --global user.email "forge-builder@noreply.github.com"

# --- gh auth ---
echo ""
echo "Authenticate gh CLI as the forge user:"
echo "  sudo -u forge gh auth login"
echo ""

# --- Telegram bot setup (optional) ---
echo "To enable Telegram control:"
echo "  1. Message @BotFather → /newbot → copy the token"
echo "  2. Send a message to the bot"
echo "  3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates → find chat_id"
echo "  4. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env"
echo ""

# --- Install systemd service ---
cp "$REPO_DIR/infra/builder/forge-builder.service" /etc/systemd/system/
systemctl daemon-reload
echo "Systemd service installed."
echo ""
echo "To start:"
echo "  1. Edit $ENV_FILE"
echo "  2. sudo -u forge gh auth login"
echo "  3. systemctl enable --now forge-builder"
echo "  4. journalctl -u forge-builder -f  (watch logs)"
echo ""
echo "=== Setup complete ==="
