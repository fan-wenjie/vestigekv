# Shared guards for pre-registered experiment scripts. Sourced, never run.
#
# Every experiment in this directory is a script, not a command someone types.
# The reason is recorded rather than assumed: hand-typed runs in this project
# have gone wrong twice in ways the shell could have caught -- an OOM from
# --gpu-expert-layers 99, and a silently wrong protocol from --rhos 0.03125
# where the flag wants a compression RATIO. A script is auditable before it
# burns GPU hours; a command in a terminal is not.

set -euo pipefail

ROOT="$HOME/vestigekv"
PY="/home/user/.conda/envs/sglang-dev/bin/python"

# WHICH ENGINE TREE. The paper's numbers were taken on the frozen tree in
# engine/ (branch vestigekv, upstream base 17ba2c2e7c). New experiments run on
# ENGINE=$HOME/vestigekv-wt/engine-v0520 (branch vestigekv-v0520, rebased onto
# upstream tag v0.5.20), which is the stable tag to build on from here.
#
# The trees are NOT interchangeable and a number from one must never be
# compared against a number from the other without saying so -- the PAT review
# already raised the frozen-vs-patched split as a concern, and this makes a
# third tree. Every script states which tree it used; the default stays the
# frozen one so nothing silently moves.
ENGINE="${ENGINE:-$ROOT/engine}"
[ -d "$ENGINE/python/sglang" ] || die_early "ENGINE has no python/sglang: $ENGINE"
export PYTHONPATH="${PYTHONPATH:-}:$ENGINE/python"

die() { echo "ABORT: $*" >&2; exit 1; }
die_early() { echo "ABORT: $*" >&2; exit 1; }

require_free_gpu() {
  # Clearing or writing results while another job holds the GPU has cost this
  # project a queue and a day; refuse rather than interleave.
  pgrep -f "[l]aunch_server" >/dev/null && die "a server is running"
  pgrep -f "[q]ueue_runner.py" >/dev/null && die "the queue runner is running"
  pgrep -f "[e]2e.py" >/dev/null && die "a harness run is in flight"
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${used:-0}" -lt 2000 ] || die "GPU still holds ${used} MiB"
}

refuse_overwrite() {
  # An experiment that silently overwrites its own record makes the archive a
  # liar. Re-running is fine; doing it by accident is not.
  [ -e "$1" ] && die "output exists: $1 (move it aside to re-run)"
  mkdir -p "$(dirname "$1")"
}

require_model_dir() {
  [ -d "$1" ] || die "model dir missing: $1 (see harness/extract.py ARCH for the rebuild recipe)"
  [ -f "$1/modeling_kimi.py" ] || die "not a plain dir (needs real *.py): $1"
}

banner() { echo "=== $* ==="; date -u +"    started %Y-%m-%dT%H:%M:%SZ"; }
