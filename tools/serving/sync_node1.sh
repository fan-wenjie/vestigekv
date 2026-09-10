#!/bin/bash
# Git-based two-machine sync (replaces rsync): the fork is the hub.
# Precondition: local vestigekv committed (the single amended commit) and
# pushed (push -f user vestigekv). Then node1 hard-resets to the fork.
set -euo pipefail
cd /home/user/fft/sglang
[ -z "$(git status --short)" ] || { echo "ABORT: local tree dirty -- amend the single commit first"; exit 1; }
TOK=$(grep -h 'set-url github https://ghp_' ~/.bash_history | tail -1 | sed 's#.*https://\(ghp_[A-Za-z0-9]*\)@.*#\1#')
git push -f user vestigekv 2>&1 | sed "s#${TOK}#***#g" | tail -1
ssh gpu5090 'G="git -C /home/user/fft/sglang"; $G fetch -q user && $G checkout -q vestigekv 2>/dev/null; $G reset -q --hard user/vestigekv && $G clean -qfd && $G log --oneline -1' 2>/dev/null
# verify: HEADs identical
L=$(git rev-parse vestigekv); R=$(ssh gpu5090 'git -C /home/user/fft/sglang rev-parse HEAD' 2>/dev/null)
[ "$L" = "$R" ] && echo "GIT SYNC OK ($L)" || { echo "SYNC MISMATCH local=$L remote=$R"; exit 1; }
