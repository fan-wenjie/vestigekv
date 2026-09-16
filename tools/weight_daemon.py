"""Step-0 weights daemon: warm, hold, and attest the checkpoint.

Reads every safetensors shard of the pinned snapshot into the OS page cache
and re-touches them periodically so later server launches (either arm) load
weights at memory speed from the SAME attested bytes. Prints the weight
fingerprint (index sha256 + shard listing sha256) once at startup; launch
scripts print the same WEIGHT-FP line, so any run's log can be checked
against the daemon's declaration.

    python tools/weight_daemon.py &          # step 0, keep running (Base checkpoint)
    python tools/weight_daemon.py --model moonshotai/Kimi-Linear-48B-A3B-Instruct &   # the Kimi RULER line
"""
import argparse
import glob
import hashlib
import os
import sys
import time

BASE_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--moonshotai--Kimi-Linear-48B-A3B-Base/"
    "snapshots/3b171c17bfc4ee348599b6781a2ca8715c21c8dc"
)
SNAP = BASE_SNAP


def snapshot_of(model):
    """The HF-cache snapshot directory of a model id (one snapshot expected;
    the pinned Base snapshot stays the default)."""
    root = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots")
    snaps = sorted(glob.glob(f"{root}/*"))
    if len(snaps) != 1:
        raise SystemExit(f"{model}: expected one snapshot under {root}, found {len(snaps)}")
    return snaps[0]


def fingerprint():
    """Byte-identical to the launch scripts' shell pipeline:
    (sha256sum index | cut -d' ' -f1; ls -l *.safetensors |
     awk '{print $5, $NF}') | sha256sum"""
    idx = hashlib.sha256(
        open(f"{SNAP}/model.safetensors.index.json", "rb").read()
    ).hexdigest()
    text = idx + "\n"
    for p in sorted(glob.glob(f"{SNAP}/*.safetensors")):
        # snapshot entries are symlinks into the content-addressed blob
        # store: the target NAME is the shard's sha256, so this line makes
        # the fingerprint content-level (identical to the launch scripts'
        # `ls -l | awk '{print $5, $NF}'` on symlinks).
        text += f"{os.lstat(p).st_size} {os.readlink(p)}\n"
    return hashlib.sha256(text.encode()).hexdigest()


def touch_all():
    n = 0
    for p in sorted(glob.glob(f"{SNAP}/*.safetensors")):
        with open(p, "rb", buffering=1 << 22) as f:
            while f.read(1 << 24):
                n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="HF model id to hold instead of the pinned Base snapshot")
    args = ap.parse_args()
    if args.model:
        SNAP = snapshot_of(args.model)
    print(f"WEIGHT-DAEMON snapshot={SNAP}", flush=True)
    print(f"WEIGHT-FP {fingerprint()}", flush=True)
    t0 = time.time()
    touch_all()
    print(f"WEIGHT-DAEMON warmed in {time.time() - t0:.0f}s; holding", flush=True)
    while True:
        time.sleep(600)
        touch_all()
        sys.stdout.write(".")
        sys.stdout.flush()
