#!/usr/bin/env bash
# Reconstruct the GLM-5.3 line's checkouts at the pinned commits, on a fresh box.
#
# Prints what it would do and asks once before touching anything. It never
# force-updates an existing checkout -- if a path exists it says so and stops, so a
# machine that already has work in progress is not clobbered.
set -euo pipefail

ROOT=${ROOT:-/home/user}
VK=$ROOT/vestigekv
WT=$ROOT/vestigekv-wt
GLM=$ROOT/vestigekv-glm

REG_SHA=6d3ce06a55b0001ae54c29f9113e52492a8c1c1b   # vestigekv  glm53-line
ENG_SHA=d3ac113872157d2903e8610fedda67c560493af2   # sglang     glm53-v0521  (SERVED)
PR_SHA=5cfc7ac455728f8e92a5296a1436967a48a2f94f    # sglang     glm53-pr

cat <<EOF
This will create, under $ROOT:
  vestigekv            (clone, submodule sglang initialised)
  vestigekv-glm        worktree of vestigekv  @ glm53-line  $REG_SHA
  vestigekv-wt/engine-glm521     worktree of the sglang SUBMODULE @ glm53-v0521
  vestigekv-wt/engine-glm521-pr  worktree of the sglang SUBMODULE @ glm53-pr

It will NOT download weights (see reference/environment.md) and will NOT create the
conda env. Nothing is force-updated.
EOF
read -r -p "proceed? [y/N] " a; [ "$a" = y ] || { echo "aborted"; exit 1; }

if [ ! -d "$VK/.git" ]; then
  git clone https://github.com/fan-wenjie/vestigekv.git "$VK"
  git -C "$VK" submodule update --init --recursive
else
  echo "exists, leaving alone: $VK"
fi

mk_wt() {  # repo, path, branch, sha
  if [ -e "$2" ]; then echo "exists, leaving alone: $2"; return; fi
  git -C "$1" fetch origin "$3"
  git -C "$1" worktree add "$2" "$4"
  git -C "$2" checkout -B "$3" "$4"
}
mk_wt "$VK" "$GLM" glm53-line "$REG_SHA"
mkdir -p "$WT"
SUB=$VK/engine
mk_wt "$SUB" "$WT/engine-glm521"    glm53-v0521 "$ENG_SHA"
mk_wt "$SUB" "$WT/engine-glm521-pr" glm53-pr    "$PR_SHA"

echo
echo "done. next: reference/environment.md for weights and the env, then"
echo "scripts/verify_env.sh before trusting any number."
