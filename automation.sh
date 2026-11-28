#!/bin/zsh
# Install, inspect, or remove the macOS daily analysis job.
set -eu
cd "${0:A:h}"
DIR="$PWD"
LABEL=com.orderbook-lab.analysis
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
mkdir -p logs

case "${1:-}" in
  install)
    [[ "$(uname -s)" == Darwin ]] || { echo 'launchd requires macOS' >&2; exit 1; }
    UV="$(command -v uv)"
    [[ -n "$UV" ]] || { echo 'uv is required' >&2; exit 1; }
    mkdir -p "$HOME/Library/LaunchAgents"
    "$UV" run --frozen python - "$PLIST" "$DIR" "$UV" <<'PY'
import plistlib
import sys

path, root, uv = sys.argv[1:]
job = {
    "Label": "com.orderbook-lab.analysis",
    "ProgramArguments": [uv, "run", "--frozen", "python", "-u", "automation.py"],
    "WorkingDirectory": root,
    "RunAtLoad": True,
    "StartCalendarInterval": {"Hour": 4, "Minute": 0},
    "StandardOutPath": f"{root}/logs/automation.log",
    "StandardErrorPath": f"{root}/logs/automation.log",
}
with open(path, "wb") as f:
    plistlib.dump(job, f)
PY
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo 'daily analysis installed for 4:00 AM local time; first run starts now'
    ;;
  run)
    uv run --frozen python -u automation.py
    ;;
  status)
    if [[ -f "$PLIST" ]]; then echo 'daily analysis installed (4:00 AM local time)'; else echo 'daily analysis not installed'; fi
    [[ -f logs/automation-status.json ]] && cat logs/automation-status.json
    [[ -f logs/automation.log ]] && tail -n 8 logs/automation.log
    ;;
  uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo 'daily analysis removed; recorder is unchanged'
    ;;
  *) echo 'usage: ./automation.sh install|run|status|uninstall' >&2; exit 1 ;;
esac
