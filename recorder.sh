#!/bin/zsh
# Start / stop / check the background recorder.
#   ./recorder.sh start | stop | status
# `caffeinate -i` keeps the Mac from idle-sleeping while the recorder runs.
# (Closing the lid on battery still sleeps a MacBook; the recorder logs that as a gap.)
cd "$(dirname "$0")"
mkdir -p logs
PIDFILE=logs/recorder.pid
case "$1" in
  start)
    if [[ -f $PIDFILE ]] && kill -0 $(cat $PIDFILE) 2>/dev/null; then echo "already running (pid $(cat $PIDFILE))"; exit 0; fi
    nohup caffeinate -i uv run python recorder.py >> logs/recorder.log 2>&1 &
    echo $! > $PIDFILE
    echo "started (pid $!), log: logs/recorder.log"
    ;;
  stop)
    if [[ -f $PIDFILE ]]; then
      # stop the python child first so it flushes its buffer, then caffeinate exits
      pkill -TERM -f "[Pp]ython recorder.py" 2>/dev/null
      for i in {1..20}; do pgrep -f "[Pp]ython recorder.py" >/dev/null || break; sleep 0.5; done
      kill $(cat $PIDFILE) 2>/dev/null; rm -f $PIDFILE; echo "stopped"
    else echo "not running"; fi
    ;;
  status)
    if [[ -f $PIDFILE ]] && kill -0 $(cat $PIDFILE) 2>/dev/null; then echo "running (pid $(cat $PIDFILE))"; else echo "not running"; fi
    du -sh data 2>/dev/null
    [[ -f data/gaps.jsonl ]] && echo "gaps logged: $(wc -l < data/gaps.jsonl)"
    tail -n 3 logs/recorder.log 2>/dev/null
    ;;
  *) echo "usage: $0 start|stop|status"; exit 1;;
esac
