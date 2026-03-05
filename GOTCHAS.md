# Builder Gotchas

## VPS Setup vs Reality

The `setup.sh` script assumes `/opt/forge` and a `forge` system user. The actual VPS uses:
- User: `admin`
- Repo: `/home/admin/forge`
- SSH: `ssh -i ~/.ssh/vps_rsa admin@89.167.63.37`

The `forge-builder.service` file shipped with the wrong user/paths and had to be fixed with sed on the server. If re-provisioning, update `setup.sh` and the service file first.

## Sudoers

`admin` has passwordless sudo via `/etc/sudoers.d/forge-deploy`. Currently wide open (`NOPASSWD: ALL`) — could be tightened to just systemctl commands.

## Hetzner VNC Console

The browser VNC console mangles special characters when typing — parens `()` become `90`, colon `:` becomes `;`. Don't use it for config files. Use Rescue Mode + SSH instead.

## Rescue Mode

If SSH (port 22) goes down: Hetzner Cloud → Rescue tab → Enable → Reboot → SSH into rescue env → `mount /dev/sda1 /mnt` → fix files → reboot.

## Auto-Deploy

GitHub Action (`.github/workflows/deploy-builder.yml`) deploys on push to `main` when `infra/builder/**` changes. Requires three GitHub secrets: `HETZNER_HOST`, `HETZNER_USER`, `HETZNER_SSH_KEY`.
