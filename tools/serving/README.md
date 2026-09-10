# Two-machine serving infrastructure (what sglang lacks: this project's hardware/topology config)

Log paths default to the session scratchpad; set LOGDIR or edit paths when
reusing.

- launch_node0/1.sh [+ _triton dense-arm variants]: paired launches with a
  cross-machine git-fingerprint sync guard (an empty fingerprint ABORTs --
  a guard that can check nothing must fail loudly). The canonical
  algorithm-arm launch is mexp/launch_vestige_server.sh (no magic numbers;
  enforced by tools/check_recommended_config.py).
- sync_node1.sh: push -f to the fork + node1 reset --hard (run before every
  launch).
- stop_servers.sh: returns only once VRAM < 2GB on both nodes (a fixed
  sleep once caused OOMs).
- verify_partition.sh: asserts the VRAM asymmetry ratio is in [2.5, 7.0],
  proving the 23,4 layer partition took effect.
- net_probe.sh: 8-stream TCP bandwidth + connect-RTT (per-launch network
  covariates).
- launch_weight_daemons.sh: resident weight daemons (pass the partition env
  explicitly).
