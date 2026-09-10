#!/bin/bash

# stale client cleanup: a perf launch must not inherit an old bench/probe
# still streaming against the previous server ([x] pattern avoids self-match)
pkill -9 -f "[s]glang.benchmark.serving" 2>/dev/null || true
pkill -9 -f "[b]ench_one_batch_server" 2>/dev/null || true
pkill -9 -f "[q]uant_ab_probe" 2>/dev/null || true
ssh gpu5090 'pkill -9 -f "[s]glang.benchmark.serving" 2>/dev/null; true' 2>/dev/null

# Stop both nodes' servers and WAIT until VRAM is actually released -- a
# fixed sleep lets the next server start while the previous one still holds
# memory, which shows up as inexplicable OOMs (observed on the 5090).
P="sglang.launch_""server"
pkill -f "$P"; ssh gpu5090 "pkill -f $P"
for i in $(seq 1 60); do
  L=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  R=$(ssh gpu5090 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1")
  [ "${L:-9999}" -lt 2000 ] && [ "${R:-9999}" -lt 2000 ] && break
  sleep 2
done
echo "VRAM released node0=${L} MiB node1=${R} MiB"
