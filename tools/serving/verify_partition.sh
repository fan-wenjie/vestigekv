#!/bin/bash
# Post-launch self-check: did the layer partition actually take effect?
# SGLANG_PP_LAYER_PARTITION is an env var -- a launcher that forgets one
# export still starts (and even runs at short context); the one piece of
# evidence that cannot hide is the VRAM asymmetry: under the 23/4 split,
# node1 (a 32GB card) carries only 4/27 of the weights. A silent even split
# on this hardware either OOMs (long context) or shows node1 holding far
# more than 4 layers' worth.
L=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
R=$(ssh gpu5090 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1")
if [ "${L:-0}" -lt 30000 ]; then echo "ABORT: node0 only ${L}MiB -- weights not loaded or server not up"; exit 1; fi
if [ "${R:-0}" -lt 5000 ]; then echo "ABORT: node1 only ${R}MiB -- weights not loaded or server not up"; exit 1; fi
if [ "$R" -gt 28000 ]; then echo "ABORT: node1 at ${R}MiB, near full -- partition likely not applied (even split)"; exit 1; fi
RATIO=$(python3 -c "print(f'{$L/$R:.2f}')")
python3 -c "exit(0 if 2.5 <= $L/$R <= 7.0 else 1)" || { echo "ABORT: VRAM ratio $RATIO outside the 23/4 split plausible band [2.5,7.0]"; exit 1; }
echo "partition self-check PASS: node0=${L}MiB node1=${R}MiB ratio $RATIO"
