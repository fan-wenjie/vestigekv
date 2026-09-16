#!/usr/bin/env bash
# Weight-cache daemon for the Kimi line: holds the TP=2-sharded bf16 weights in GPU
# memory (one daemon process per GPU, sglang.srt.weight_cache.daemon) so that every
# server the queue launches loads them over CUDA IPC in seconds instead of from disk.
# The arm scripts (common.sh) switch to --weight-cache-mode client automatically when
# the daemon's ready files exist; the KV pool is sized from what the daemon leaves,
# exactly as mem-fraction sizing did before.
#   nohup bash mexp/kimi/weight_daemon.sh > results/kimi/weight_daemon.log 2>&1 &   # start (foreground otherwise)
#   bash mexp/kimi/weight_daemon.sh status                                            # ready files and pids
#   bash mexp/kimi/weight_daemon.sh stop                                              # SIGTERM the daemon group
# The cache config (model path, TP=2, dtype, MoE layout, torch and GPU stamp) must match
# the server's; on any mismatch the engine falls back to disk loading and logs it, so
# every server log is checked for "[IpcModelLoader] Loaded model via IPC".
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
case "${1:-start}" in
  status)
    for f in /tmp/sglang_weight_cache_*.ready; do [ -e "$f" ] && echo "$f: $(tr '\n' ' ' < "$f")"; done
    pgrep -af "^[^ ]*python[^ ]* -m sglang.srt.weight_cache.daemon" | cut -c1-120 ;;
  stop)
    pkill -TERM -f "^[^ ]*python[^ ]* -m sglang.srt.weight_cache.daemon"; sleep 5
    pgrep -af "weight_cache" | cut -c1-120 || echo "daemon stopped" ;;
  start)
    if pgrep -f "^[^ ]*python[^ ]* -m sglang.launch_server" > /dev/null; then echo "ABORT: a server is running"; exit 1; fi
    cd "$ROOT" && exec $PY -m sglang.srt.weight_cache.daemon --model-path "$MODEL" --trust-remote-code \
      --tp-size 2 --disable-custom-all-reduce "${@:2}" ;;
  *) echo "usage: $0 [start|status|stop]"; exit 2 ;;
esac
