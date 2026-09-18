#!/usr/bin/env bash
# Experiment health monitor: one line per interval in results/<line>/health.log.
#   usage: mexp/health_check.sh <line> [interval_s]      (default 1800; runs until killed)
# Reports: queue runner / server liveness, current job, client progress and whether it
# moved since the last check (STALL otherwise), GPU memory and utilization, errors seen
# in the current job's server log since the last check, free disk. Read-only.
LINE=${1:?line: kimi | glm53}; EVERY=${2:-1800}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R=$ROOT/results/$LINE; LOG=$R/health.log; mkdir -p "$R"
prev_prog=""; prev_err=0; prev_slog=""
while true; do
  now=$(date +%F_%T)
  # anchored at the python executable, so a shell whose command text quotes these commands
  # (a restart watcher) is not counted
  runner=$(pgrep -fc "^[^ ]*python[^ ]* mexp/glm53/queue_runner.py --line $LINE"); [ "$LINE" = glm53 ] && runner=$(pgrep -fc "^[^ ]*python[^ ]* mexp/glm53/queue_runner.py")
  servers=$(pgrep -fc "^[^ ]*python[^ ]* -m sglang.launch_server")
  daemon=$(pgrep -fc "^[^ ]*python[^ ]* -m sglang.srt.weight_cache.daemon")
  job=$(grep -a '"status": "running"' "$R/queue_state.jsonl" 2>/dev/null | tail -1 | python3 -c 'import sys,json; l=sys.stdin.read().strip(); print(json.loads(l)["id"] if l else "-")')
  done_n=$(grep -ac '"status": "done"' "$R/queue_state.jsonl" 2>/dev/null); fail_n=$(grep -ac '"status": "failed"' "$R/queue_state.jsonl" 2>/dev/null)
  slog=$(ls -t "$R"/server_*.log 2>/dev/null | head -1)
  clog=$(ls -t "$R"/ruler_*_*.log "$R"/stream_*_*.log "$R"/replay_*_*.log "$R"/longbench2_*_*.log "$R"/continue_*_*.log 2>/dev/null | head -1)
  # a bare tqdm bar counts as progress too: run_longbench2.py writes one with no
  # description, and matching only the labelled clients reported stale progress
  # and a STALL on every check while it ran.
  # continue_text.py prints a bare "  4/8  123s" with no tqdm bracket, so a
  # pattern anchored on "[" saw no progress for the whole of a litspeed job and
  # would have reported STALL on every check while it ran correctly.
  prog=$( [ -n "$clog" ] && tr '\r' '\n' < "$clog" | grep -aE "Requesting API|Prefill batch|== RULER|[0-9]+/[0-9]+ \[|[0-9]+/[0-9]+  [0-9]+s" | tail -1 | grep -oE "[0-9]+/[0-9]+ \[[^]]*\]|[0-9]+/[0-9]+  [0-9]+s|== RULER.*" | head -1 )
  # a stream job's bar reads 0/1 for its whole run (one request), so the decode
  # token count in the server log is its only progress signal
  case "$clog" in *"/stream_"*) prog=$(grep -a "Decode batch" "$slog" 2>/dev/null | tail -1 | grep -oE "#(full )?token: [0-9]+" | head -1);; esac
  stall=""; [ -n "$prog" ] && [ "$prog" = "$prev_prog" ] && [ "$job" != "-" ] && stall=" STALL(no client progress since last check)"
  prev_prog=$prog
  # tracebacks in the newest server log, counted per file: a switch to a new log restarts
  # the count instead of differencing against the previous log's total
  [ "$slog" != "$prev_slog" ] && prev_err=0; prev_slog=$slog
  # Anchored on real failures. "CUDA error" unanchored matched sglang's own
  # benign context-length warning, whose text ends "...or CUDA errors.", so every
  # long-context job (CTX above the model's derived length, which is deliberate
  # and passed with SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN) reported
  # SERVER_ERRORS while running correctly.
  err=$( [ -n "$slog" ] && grep -ac "Scheduler hit an exception\|OutOfMemoryError\|CUDA error:\|^Traceback\|^\[[^]]*\] Traceback" "$slog" )
  new_err=$(( ${err:-0} - prev_err )); prev_err=${err:-0}
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk '{printf "%s/%s%% ", $1, $2}' | tr -d ',')
  disk=$(df -h "$ROOT" | awk 'NR==2{print $4}')
  status=OK; { [ "$runner" = 0 ] && [ "$job" != "-" ]; } && status=RUNNER_DEAD
  [ "$new_err" -gt 0 ] && status=SERVER_ERRORS; [ -n "$stall" ] && status=STALL; [ "$runner" = 0 ] && [ "$job" = "-" ] && status=IDLE
  echo "$now $status runner=$runner servers=$servers daemon=$daemon job=$job done=$done_n failed=$fail_n prog='${prog:-}' new_err=$new_err gpu=$gpu disk_free=$disk$stall" >> "$LOG"
  sleep "$EVERY"
done
