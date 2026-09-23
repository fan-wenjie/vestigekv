#!/usr/bin/env bash
# Reproduce the intermittent garbage answers: launch the vestigekv arm with extra env
# (e.g. SGLANG_DEBUG_VESTIGEKV_ROWS=1, SGLANG_ENABLE_VESTIGEKV_INGRAPH_SCAN=0, GRAPH=0),
# run all 13 RULER tasks at n samples/cell, count answers with the '、}' repetition
# pattern, tear down. Appends "<tag> garbage=k/n" to results/kimi/garbage_probe.log.
#   usage: mexp/kimi/garbage_probe.sh <tag> [n=4] [extra server args...]   (env passes through)
set -euo pipefail
TAG=${1:?tag}; N=${2:-4}; shift 2 || shift $#
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PYTHON:-$HOME/.conda/envs/sglang-dev/bin/python}
OUT=$ROOT/results/kimi/debug; mkdir -p "$OUT"
cd "$ROOT"
setsid nohup bash mexp/kimi/vestigekv.sh "$@" > "$OUT/server_probe_$TAG.log" 2>&1 &
until grep -qa "fired up\|Scheduler hit\|ABORT" "$OUT/server_probe_$TAG.log"; do sleep 10; done
grep -qa "fired up" "$OUT/server_probe_$TAG.log" || { echo "$TAG server failed"; exit 1; }
CUDA_VISIBLE_DEVICES="" $PY mexp/exp/run_ruler.py --arm "probe-$TAG" --port 30000 \
  --model moonshotai/Kimi-Linear-48B-A3B-Instruct --n "$N" --out "$OUT" > "$OUT/ruler_probe_$TAG.log" 2>&1 || true
for p in $(pgrep -f "^$PY -m sglang[.]launch_server"); do kill -TERM "$p"; done
line=$($PY - "$OUT/samples_probe-${TAG}_n${N}_4096-8192-16384-32768-65536.json" "$TAG" <<'PYEOF'
import json, re, sys
s = json.load(open(sys.argv[1])); rows = []
for task, docs in s.items():
    for d in docs:
        r = d["resps"][0][0]
        g = bool(re.search(r"(、\}|\}\)|、、|\)\))", r)) or (len(r) > 20 and len(set(r)) < 8)
        rows.append((len(d["doc"]["input"]), g))
rows.sort(key=lambda x: -x[0])
print(f"{sys.argv[2]} garbage={sum(g for _, g in rows)}/{len(rows)} order={''.join('G' if g else '.' for _, g in rows)}")
PYEOF
)
echo "$(date +%F_%T) $line" | tee -a "$ROOT/results/kimi/garbage_probe.log"
sleep 15
