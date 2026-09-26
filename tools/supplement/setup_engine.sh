#!/usr/bin/env bash
# Rebuild the serving engine used for every measurement in the paper:
# upstream sglang at a pinned tag, plus the VestigeKV patch.
#
# The archive ships no vendored engine. It ships this script and one patch, so
# what you build is verifiably upstream code plus a diff you can read
# (engine/vestigekv.patch; its statistics are in engine/README.md).
#
#   ./setup_engine.sh [target-dir]     # default: ./sglang
#
# The engine is NOT installed. The harness runs it from source via PYTHONPATH;
# installing it would shadow the tree you just patched. The last lines printed
# tell you what to export.

set -euo pipefail

TAG="v0.5.20"
TARBALL_URL="https://github.com/sgl-project/sglang/archive/refs/tags/${TAG}.tar.gz"
PATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vestigekv.patch"
TARGET="${1:-$(pwd)/sglang}"

[ -f "$PATCH" ] || { echo "patch not found: $PATCH" >&2; exit 1; }

if [ -e "$TARGET" ]; then
  echo "target already exists: $TARGET"
  echo "remove it or pass another directory; refusing to patch over unknown state." >&2
  exit 1
fi

echo "==> fetching sglang $TAG source tarball"
# A release tarball, not a clone: no history is needed, and this works where
# git access to github.com is proxied but plain HTTPS is not.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fL "$TARBALL_URL" -o "$TMP/sglang.tar.gz"
tar -xzf "$TMP/sglang.tar.gz" -C "$TMP"
mv "$TMP/sglang-${TAG#v}" "$TARGET"

echo "==> checking the patch applies cleanly"
patch -d "$TARGET" -p1 --dry-run < "$PATCH"

echo "==> applying"
patch -d "$TARGET" -p1 < "$PATCH"

cat <<EOF

Done. The engine is at:
    $TARGET

Run it from source -- do not pip install it, which would shadow this tree:
    export PYTHONPATH="$TARGET/python:\$PYTHONPATH"
    python -c "import sglang; print(sglang.__file__)"

The attention backend registers as --attention-backend vestigekv_mla.
See EXPERIMENTS.md for the exact launch commands behind each number.
EOF
