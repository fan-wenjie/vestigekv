#!/usr/bin/env bash
# Preflight. Run before trusting a measurement on a new box; each check
# corresponds to something that silently produced a wrong number here.
PY=${PY:-/home/user/.conda/envs/sglang-dev/bin/python}
ENG=${ENG:-/home/user/vestigekv-wt/engine-glm521}
fail=0
ck() { if [ "$2" = ok ]; then echo "  ok   $1"; else echo "  FAIL $1 -- $3"; fail=1; fi; }

echo "=== arch"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
[ "$cc" = "12.0" ] && ck "compute cap $cc (SM120)" ok || ck "compute cap $cc" bad \
  "measured on 12.0; trtllm/flashmla have no FMHA kernel there so the triton-only patches may not apply"
n=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[ "$n" -ge 2 ] && ck "$n GPUs" ok || ck "$n GPU" bad "the line serves tp-size 2"

echo "=== toolchain"
[ -x /usr/local/cuda/bin/nvcc ] && ck "nvcc at /usr/local/cuda" ok || ck "nvcc" bad \
  "CUDA_HOME must be /usr/local/cuda or the JIT dies with 'cicc: not found'"
$PY -c "import torch" 2>/dev/null && ck "torch imports" ok || ck "torch" bad "env missing"

echo "=== sglang comes from the engine tree, never an install"
$PY - <<'P' 2>/dev/null
import importlib.util, os, sys
s = importlib.util.find_spec("sglang")
p = s.origin if s else None
want = os.environ.get("ENG", "/home/user/vestigekv-wt/engine-glm521")
print("ok" if p and p.startswith(want) else f"BAD {p}")
P
echo "     (expect ok; a BAD path means an installed sglang is shadowing the tree)"

echo "=== pinned commits"
for d in /home/user/vestigekv-glm $ENG; do
  [ -d "$d" ] && echo "  $(basename $d): $(git -C $d rev-parse --abbrev-ref HEAD) $(git -C $d rev-parse --short HEAD) dirty=$(git -C $d status --porcelain | wc -l)"
done

echo "=== weights"
for m in models--nvidia--GLM-5.3-Flash-NVFP4; do
  d=$HOME/.cache/huggingface/hub/$m
  [ -d "$d" ] && ck "$m ($(du -sh $d | cut -f1))" ok || ck "$m" bad "hf download nvidia/GLM-5.3-Flash-NVFP4"
done

echo "=== nothing already holding the GPU"
u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
[ "$u" -lt 1000 ] && ck "GPU idle (${u} MiB)" ok || ck "GPU busy (${u} MiB)" bad \
  "a runner may be live; never edit the served tree while one is"

echo
[ $fail -eq 0 ] && echo "PREFLIGHT OK" || echo "PREFLIGHT HAS FAILURES -- read the notes above"
exit $fail
