#!/usr/bin/env bash
# Rebuild the serving engine used for every measurement in the paper:
# upstream sglang at a pinned commit, plus the VestigeKV patch.
#
# The archive ships no vendored engine. It ships this script and one patch, so
# what you build is verifiably upstream code plus a diff you can read
# (engine/vestigekv.patch, 11105 added lines across 30 files, none deleted:
# outside its own package the change is a single insertion hunk per file).
#
#   ./setup_engine.sh [target-dir]     # default: ./sglang
#
# The engine is NOT installed. The harness runs it from source via PYTHONPATH;
# installing it would shadow the tree you just patched. The last lines printed
# tell you what to export.

set -euo pipefail

UPSTREAM_URL="https://github.com/sgl-project/sglang.git"
UPSTREAM_BASE="94602c9c2b7cbdb8efd5c52802dac6a1c180089e"  # the v0.5.20 tag
PATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vestigekv.patch"
TARGET="${1:-$(pwd)/sglang}"

[ -f "$PATCH" ] || { echo "patch not found: $PATCH" >&2; exit 1; }

if [ -e "$TARGET" ]; then
  echo "target already exists: $TARGET" >&2
  echo "remove it or pass another directory; refusing to patch over unknown state." >&2
  exit 1
fi

echo "==> fetching sglang $UPSTREAM_BASE from $UPSTREAM_URL"
# A blobless partial clone: the full history of this repository is large and
# none of it is needed except one commit's tree.
git clone --filter=blob:none --no-checkout "$UPSTREAM_URL" "$TARGET"
git -C "$TARGET" fetch --depth 1 origin "$UPSTREAM_BASE"
git -C "$TARGET" checkout --detach "$UPSTREAM_BASE"

echo "==> verifying the base commit"
have="$(git -C "$TARGET" rev-parse HEAD)"
[ "$have" = "$UPSTREAM_BASE" ] || { echo "base mismatch: $have" >&2; exit 1; }
echo "    $have  $(git -C "$TARGET" log -1 --format=%s)"

echo "==> checking the patch applies cleanly"
git -C "$TARGET" apply --check --whitespace=nowarn "$PATCH"

echo "==> applying"
git -C "$TARGET" apply --whitespace=nowarn "$PATCH"

echo "==> result"
git -C "$TARGET" -c core.fileMode=false status --porcelain | awk '{print "    " $0}' | head -40
n=$(git -C "$TARGET" status --porcelain | wc -l)
echo "    $n files changed"

cat <<EOF

Done. The engine is at:
    $TARGET

Run it from source -- do not pip install it, which would shadow this tree:
    export PYTHONPATH="$TARGET/python:\$PYTHONPATH"
    python -c "import sglang; print(sglang.__file__)"

The attention backend registers as --attention-backend vestigekv_mla.
See EXPERIMENTS.md for the exact launch commands behind each number.
EOF
