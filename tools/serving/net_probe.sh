#!/bin/bash
# Network-state covariate: 8 parallel raw TCP streams x 64MB (512MB total),
# aggregate MB/s. Matches NCCL's multi-socket transfer shape; a single
# stream pins at 148MB/s and carries no information.
for p in $(seq 19870 19877); do
  ssh gpu5090 "nohup timeout 40 nc -l -p $p > /dev/null 2>&1 &" 2>/dev/null
done
sleep 1
S=$(date +%s.%N)
for p in $(seq 19870 19877); do
  (dd if=/dev/zero bs=4M count=16 2>/dev/null | timeout 35 nc -q1 172.20.19.32 $p) &
done
wait
E=$(date +%s.%N)
python3 -c "print(f'{512/($E-$S):.0f}')"
# Latency covariate: bs=1 decode moves ~4.6KB per NCCL hop -- a
# latency-bound load. No ping in the container; the median of 20 TCP
# connect times (SYN round-trip) approximates RTT (ms).
python3 - <<'PYEOF'
import socket, time
ts = []
for _ in range(20):
    t0 = time.perf_counter()
    s = socket.create_connection(("172.20.19.32", 22), timeout=5)
    ts.append((time.perf_counter() - t0) * 1e3)
    s.close()
ts.sort()
print(f"{ts[len(ts)//2]:.2f}")
PYEOF
