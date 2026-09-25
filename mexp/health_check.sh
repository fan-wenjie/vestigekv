#!/usr/bin/env bash
# Experiment health monitor: one line per interval in results/<line>/health.log.
#   usage: mexp/health_check.sh <line> [interval_s]      (default 1800; runs until killed)
# Reports: queue runner / server liveness, current job, client progress and whether it
# moved since the last check (STALL otherwise), GPU memory and utilization, errors seen
# in the current job's server log since the last check, free disk. Read-only.
LINE=${1:?line: kimi}; EVERY=${2:-1800}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R=$ROOT/results/$LINE; LOG=$R/health.log; mkdir -p "$R"
prev_prog=""; prev_err=0; prev_slog=""
# The models THIS line can serve: common.sh's default plus any MODEL a job
# overrides in its env. Computed once, because the alternative is matching every
# sglang server on the box -- which is what made a drained Kimi line report a
# server and 87% GPU that belonged to the GLM run on the other tree.
MODELS=$(python3 - "$ROOT" "$LINE" <<'PY'
import json, os, re, sys
root, line = sys.argv[1], sys.argv[2]
ms = set()
common = os.path.join(root, "mexp", line, "common.sh")
if os.path.exists(common):
    m = re.search(r"^MODEL=\$\{MODEL:-(.+?)\}|^MODEL=(.+)$", open(common).read(), re.M)
    if m:
        ms.add((m.group(1) or m.group(2)).strip())
q = os.path.join(root, "mexp", line, "queue.jsonl")
if os.path.exists(q):
    for l in open(q):
        if l.strip():
            ms.add((json.loads(l).get("env") or {}).get("MODEL", ""))
print("|".join(re.escape(x) for x in sorted(ms) if x))
PY
)
while true; do
  now=$(date +%F_%T)
  # anchored at the python executable, so a shell whose command text quotes these commands
  # (a restart watcher) is not counted
  # The runner always passes --line, so matching on it is the whole answer.
  # Dropping it let a monitor count another line's runner as its own.
  runner=$(pgrep -fc "^[^ ]*python[^ ]* mexp/exp/queue_runner.py --line $LINE")
  # Scoped to this line's models (see MODELS above). Unscoped it counted every
  # sglang server on the box; with no model resolvable, fall back to that rather
  # than silently report zero servers while one is up.
  # NOT `scoped || unscoped`: pgrep -c exits 1 when the count is zero, so the
  # fallback ran whenever this line had no server up and $servers became the two
  # lines "0" and "1" -- which broke the log record in half AND defeated the
  # scoping it was there to provide. The count of zero is an answer, not a
  # failure, so the branch is on whether a model resolved at all.
  if [ -n "$MODELS" ]; then
    servers=$(pgrep -fc "^[^ ]*python[^ ]* -m sglang.launch_server .*--model-path (${MODELS})( |$)" || true)
  else
    servers=$(pgrep -fc "^[^ ]*python[^ ]* -m sglang.launch_server" || true)
  fi
  daemon=$(pgrep -fc "^[^ ]*python[^ ]* -m sglang.srt.weight_cache.daemon")
  # The LAST row, and only if it is a running one. Taking the last "running" row
  # anywhere in the file meant every job that had ever started still counted as
  # running: the runner appends "running" and later "done", so the most recent
  # "running" line is the last job forever. A drained queue therefore never
  # reached IDLE and reported STALL on every check instead, because prog had
  # stopped moving for the entirely correct reason that the client had exited.
  # It also resurrected a stale row -- gauss-0999-ruler, started 2026-09-18,
  # never terminated, retired out of queue.jsonl since -- which only the
  # last-row rule excludes, a killed runner leaving its row behind. A live job
  # is always the file's tail; a leftover never is, and a leftover that IS the
  # tail is a dead runner, which RUNNER_DEAD below then says.
  job=$(tail -1 "$R/queue_state.jsonl" 2>/dev/null | python3 -c 'import sys,json; l=sys.stdin.read().strip(); r=json.loads(l) if l else {}; print(r.get("id","-") if r.get("status")=="running" else "-")')
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
