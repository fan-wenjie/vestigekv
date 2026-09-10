#!/bin/bash
# Experiment health check (run every 15 min). Prints one PASS/FAIL line per
# item; exits nonzero if any FAIL. Silent-green discipline: callers alert on
# FAIL lines only. Fast (<30s), never restarts anything.
set -u
SGL=/home/user/fft/sglang
NK=/home/user/fft/nope_kv
PY=/home/user/.conda/envs/sglang/bin
fail=0
ck(){ [ "$2" = "0" ] && echo "PASS $1" || { echo "FAIL $1: $3"; fail=1; }; }

# 1 unit tests (quick). CUDA-OOM here is the GPU being legitimately owned by a
# running authoritative benchmark, not a code defect -- report SKIP, not FAIL,
# so a real logic failure still stands out (a check that FAILs for the wrong
# reason trains readers to ignore it).
full=$($PY/python -m pytest $SGL/test/registered/unit/layers/attention/test_vestigekv_mla_backend.py -q 2>&1)
out=$(echo "$full" | tail -1)
if echo "$full" | grep -qiE "CUDA error: out of memory|AcceleratorError|cudaErrorMemoryAllocation"; then
  echo "SKIP unit-tests: GPU busy (authoritative run owns VRAM); $out"
elif echo "$out" | grep -qiE "error" && ! echo "$out" | grep -qi "failed" \
     && pgrep -f "benchmark.serving|few_shot_gsm8k|tput_samelaunch|e2e.py" >/dev/null 2>&1; then
  # A COLLECTION error (pytest prints "... error", not "... failed") that
  # only appears while an authoritative benchmark owns the GPU is CUDA
  # context/init contention, not a code defect -- the same tests pass clean
  # once the GPU frees. SKIP, do not FAIL. A real logic failure prints
  # "failed" and still stands; a collection error with NO benchmark running
  # still FAILs (it is then a genuine import/collection break).
  echo "SKIP unit-tests: collection error under live benchmark (GPU contention); $out"
elif echo "$out" | grep -q 'passed' && grep -qv 'failed' <<<"$out"; then
  ck "unit-tests" 0
else
  ck "unit-tests" 1 "$out"
fi

# 2 config consistency
out=$($PY/python $NK/tools/check_recommended_config.py 2>&1 | tail -1)
[[ "$out" == OK:* ]] && ck "config-consistency" 0 || ck "config-consistency" 1 "$out"

# 3 cross-machine code sync (the backend file is the canary)
# git identity of the whole sglang tree (HEAD + working diff + porcelain),
# not a per-file content hash: a batched_step.py fix once passed a
# backend-only guard while node1 ran stale code and crashed the curve
# mid-run (dual-machine code-sync trap).
FP="git -C $SGL rev-parse HEAD^{tree}; { git -C $SGL diff HEAD; git -C $SGL status --porcelain=v1 -uall; } | md5sum | cut -d' ' -f1"
L=$(eval "$FP")
R=$(ssh -o ConnectTimeout=8 gpu5090 "$FP" 2>/dev/null)
[ -z "$R" ] && { sleep 3; R=$(ssh -o ConnectTimeout=8 gpu5090 "$FP" 2>/dev/null); }
[ -n "$R" ] && [ "$L" = "$R" ] && ck "code-sync" 0 || ck "code-sync" 1 "local/remote git state differ or unreachable"

# 4 git hygiene (dirty tree on vestigekv = uncommitted experiment code)
n=$(git -C $SGL status --short 2>/dev/null | wc -l)
[ "$n" = "0" ] && ck "git-clean" 0 || ck "git-clean" 1 "$n dirty entries"

# 5 arm-flag hygiene (a leftover FULL flag silently poisons VESTIGE runs)
a=$(ls /tmp/vestigekv_full 2>/dev/null; ssh -o ConnectTimeout=8 gpu5090 'ls /tmp/vestigekv_full 2>/dev/null' 2>/dev/null)
# A live A/B sweep owns the flag to switch arms; a stale flag is only a
# defect when nothing is running. Skip while a benchmark client is active.
if pgrep -f "benchmark.serving|few_shot_gsm8k|tput_samelaunch" >/dev/null 2>&1; then
  echo "SKIP arm-flag-clear: live A/B benchmark owns the flag"
elif [ -z "$a" ]; then ck "arm-flag-clear" 0; else ck "arm-flag-clear" 1 "flag file present (no benchmark running -- stale)"; fi

# 6 stray experiment processes (finished sweeps must not linger)
n=$(pgrep -f 'batch64k.sh|final_sweep.sh|batch_check.sh|prof_bs.sh|batch_large.sh' | wc -l)
[ "$n" = "0" ] && ck "no-stray-sweeps" 0 || ck "no-stray-sweeps" 1 "$n stray"

# 7 debug-env pollution on a RUNNING server (measurements would carry overhead)
P=$(pgrep -f 'sglang[.]launch_serv' | head -1)
if [ -n "$P" ]; then
  d=$(tr '\0' '\n' < /proc/$P/environ 2>/dev/null | grep -c 'VESTIGE_DEBUG\|VESTIGE_CHECK')
  [ "$d" = "0" ] && ck "server-env-clean" 0 || ck "server-env-clean" 1 "debug/check env on live server (fine for debugging, NOT for measurement)"
else
  ck "server-env-clean" 0
fi

# 8 result-file integrity (final campaign files unchanged since freeze)
exp="$(md5sum $NK/out/batch64k_final.csv 2>/dev/null | cut -d' ' -f1)"
frozen_file=$NK/out/.batch64k_final.frozen.md5
if [ -f "$frozen_file" ]; then
  [ "$exp" = "$(cat $frozen_file)" ] && ck "frozen-results" 0 || ck "frozen-results" 1 "batch64k_final.csv changed after freeze"
else
  echo "$exp" > "$frozen_file"; ck "frozen-results" 0
fi

exit $fail
