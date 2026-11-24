#!/bin/zsh
# Start / stop / check the background recorder.
#
#   ./recorder.sh start       start now (until stopped, logout or reboot)
#   ./recorder.sh stop        stop cleanly (buffered rows are written first)
#   ./recorder.sh status      running?, disk used, gaps, last log lines
#   ./recorder.sh install     always-on mode: start at login, restart after a crash
#   ./recorder.sh uninstall   turn always-on mode off
#
# `caffeinate -i` keeps the Mac from idle-sleeping while the recorder runs.
# (Closing a MacBook's lid still sleeps it; the recorder logs that as a gap.)
cd "$(dirname "$0")"
DIR="$(pwd)"
mkdir -p logs
PIDFILE=logs/recorder.pid
LABEL=com.orderbook-lab.recorder
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
PAT='^[^ ]*[Pp]ython[0-9.]* recorder\.py'   # the Python process only, not its caffeinate/uv wrappers
running() { pgrep -f "$PAT" >/dev/null; }
installed() { [[ -f $PLIST ]]; }

case "$1" in
  start)
    if running; then echo "already running"; exit 0; fi
    if installed; then
      launchctl kickstart "$DOMAIN/$LABEL" && echo "started (always-on mode), log: logs/recorder.log"
    else
      nohup caffeinate -i uv run python recorder.py >> logs/recorder.log 2>&1 &
      echo $! > $PIDFILE
      echo "started (pid $!), log: logs/recorder.log"
    fi
    ;;
  stop)
    if running; then
      # SIGTERM makes the recorder write its buffer and exit cleanly. In always-on mode a
      # clean exit is not restarted, so it stays stopped until `start` or the next login.
      pkill -TERM -f "$PAT"
      for i in {1..20}; do running || break; sleep 0.5; done
      [[ -f $PIDFILE ]] && { kill $(cat $PIDFILE) 2>/dev/null; rm -f $PIDFILE; }
      echo "stopped"
    else echo "not running"; fi
    ;;
  status)
    if running; then echo "running$(installed && echo ' (always-on mode)')"; else echo "not running$(installed && echo ' (always-on mode installed)')"; fi
    du -sh data 2>/dev/null
    [[ -f data/gaps.jsonl ]] && echo "gaps logged: $(wc -l < data/gaps.jsonl | tr -d ' ')"
    tail -n 3 logs/recorder.log 2>/dev/null
    ;;
  install)
    UV="$(command -v uv)"
    [[ -z $UV ]] && { echo "uv not found; install it first"; exit 1; }
    if running && ! installed; then "$0" stop; fi
    mkdir -p "$HOME/Library/LaunchAgents"
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/caffeinate</string><string>-i</string>
    <string>$UV</string><string>run</string><string>python</string><string>recorder.py</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$DIR/logs/recorder.log</string>
  <key>StandardErrorPath</key><string>$DIR/logs/recorder.log</string>
</dict>
</plist>
EOF
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
    launchctl bootstrap "$DOMAIN" "$PLIST" && echo "always-on mode installed: starts at login, restarts after a crash"
    ;;
  uninstall)
    if installed; then
      launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
      rm -f "$PLIST"; echo "always-on mode removed (recorder stopped)"
    else echo "always-on mode was not installed"; fi
    ;;
  *) echo "usage: $0 start|stop|status|install|uninstall"; exit 1;;
esac
