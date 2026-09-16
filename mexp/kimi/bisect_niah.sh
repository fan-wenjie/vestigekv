#!/usr/bin/env bash
# Serve one engine commit on the vestigekv arm and count garbage needle answers.
#   usage: mexp/kimi/bisect_niah.sh <engine-commit> [n_per_cell=10] [tasks=niah_single_1,niah_single_2]
# Checks the commit out in a throwaway worktree, launches mexp/kimi/vestigekv.sh with
# ENGINE=<worktree>, runs the RULER client on the needle tasks at 4k-64k, then prints
# "<commit> garbage=<k>/<n> mean=<score>" (garbage = answers with the '、}' repetition
# pattern or fewer than 8 distinct characters) and tears everything down. One line is
# appended to results/kimi/bisect.log.
set -euo pipefail
C=${1:?engine commit}; N=${2:-10}; TASKS=${3:-niah_single_1,niah_single_2}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PYTHON:-$HOME/.conda/envs/sglang-dev/bin/python}
WT=$HOME/vestigekv-wt/bisect-$C
OUT=$ROOT/results/kimi/bisect; mkdir -p "$OUT"
cd "$ROOT/engine" && git worktree add -q --detach "$WT" "$C"
cd "$ROOT"
ENGINE=$WT setsid nohup bash mexp/kimi/vestigekv.sh > "$OUT/server_$C.log" 2>&1 &
until grep -qa "fired up\|Scheduler hit\|ABORT" "$OUT/server_$C.log"; do sleep 10; done
if ! grep -qa "fired up" "$OUT/server_$C.log"; then echo "$C server failed"; exit 1; fi
CUDA_VISIBLE_DEVICES="" $PY mexp/glm53/run_ruler.py --arm "bisect-$C" --port 30000 \
  --model moonshotai/Kimi-Linear-48B-A3B-Instruct --n "$N" --tasks "$TASKS" --out "$OUT" > "$OUT/ruler_$C.log" 2>&1 || true
for p in $(pgrep -f "^$PY -m sglang[.]launch_server"); do kill -TERM "$p"; done
line=$($PY - "$OUT/samples_bisect-${C}_n${N}_4096-8192-16384-32768-65536.json" "$C" <<'PYEOF'
import json, re, sys
s = json.load(open(sys.argv[1])); n = g = 0; hit = 0
for task, docs in s.items():
    for d in docs:
        r = d["resps"][0][0]; n += 1
        g += bool(re.search(r"(、\}|\}\)|、、|\)\))", r)) or (len(r) > 20 and len(set(r)) < 8)
        tgt = [str(t) for t in d["doc"].get("outputs", [])]
        hit += any(t.lower() in r.lower() for t in tgt)
print(f"{sys.argv[2]} garbage={g}/{n} hit={hit}/{n}")
PYEOF
)
echo "$(date +%F_%T) $line" | tee -a "$ROOT/results/kimi/bisect.log"
sleep 15; cd "$ROOT/engine" && git worktree remove --force "$WT"
