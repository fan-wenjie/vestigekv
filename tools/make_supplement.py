#!/usr/bin/env python3
"""Build the anonymous supplementary archive for a double-blind submission.

The archive is not the repository. Three things differ, each for a reason:

1. **No git metadata.** `.git`, `.gitmodules` and `.gitattributes` are dropped.
   A repository carries author names and email addresses in every commit, and a
   submodule URL names the author's account, so shipping history would break
   anonymity however clean the working tree is.

2. **No vendored engine.** The serving engine is upstream sglang plus one
   additive change. Vendoring a fork of a large project into the archive is both
   enormous and identifying, so the archive carries `engine/vestigekv.patch` --
   the diff against a pinned upstream commit -- and `engine/setup_engine.sh`,
   which fetches that commit from the official repository and applies it. The
   patch is produced with `git diff`, not `git format-patch`, so it has no
   author header to begin with.

3. **Redacted paths and identifiers.** Absolute home paths name a user. They are
   rewritten to `$HOME`, which also keeps the scripts runnable.

The build fails rather than ships if any identifying pattern survives: `--verify`
rescans the finished tree, including PDF text, and a single hit is a hard error.

    python tools/make_supplement.py --out /tmp/supplement --zip
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Upstream base the patch applies to.
UPSTREAM_URL = "https://github.com/sgl-project/sglang.git"
UPSTREAM_BASE = "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"  # v0.5.20
ENGINE_BRANCH = "origin/vestigekv"

# --- what goes in -----------------------------------------------------------
# Directory or file -> why it is here. Anything not listed is left out, so a new
# directory has to be considered rather than swept in.
INCLUDE = {
    "EXPERIMENTS.md": "every experiment command, registered before it ran",
    "ENV_PINS.txt": "pinned environment",
    "config": "serving configuration",
    "harness": "the measurement harness and its gates",
    "mexp": "experiment drivers, pre-registrations and the macro generators",
    "tools": "utilities, including this script and the stream reducer",
    "docs": "technical notes the paper points to",
    "results": "the reduced run records (results.zip) and the serving figures",
}
# Excluded with a reason, so the list is auditable rather than arbitrary.
EXCLUDE_TREES = {
    ".git": "history carries author identity",
    ".gitmodules": "submodule URL names the author's account",
    ".gitattributes": "git metadata",
    "engine": "replaced by engine/vestigekv.patch + setup_engine.sh",
    "skills": "assistant workflow rules, not experiment code",
    "runs": "raw server logs; the run record is EXPERIMENTS.md and results.zip",
}
EXCLUDE_GLOBS = ("__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".DS_Store")
# This script and its templates cannot be in the archive it builds: a redaction
# rule has to spell out the string it redacts, so the file that removes the
# author's name is the one file guaranteed to contain it. The identity scan
# catches this if the exclusion is ever removed -- it did, which is why the rule
# is here rather than in a comment somewhere.
EXCLUDE_FILES = {
    os.path.join("tools", "make_supplement.py"),
    os.path.join("tools", "supplement"),
}

# --- redaction --------------------------------------------------------------
# Ordered: longest first, so a specific path is not half-rewritten by a general
# rule. Values keep the scripts runnable where they can.
REDACTIONS = [
    (re.compile(r"/home/[A-Za-z0-9_.-]+"), "$HOME"),
    (re.compile(r"TODO\(fan-wenjie\)", re.I), "TODO"),
    (re.compile(r"https://github\.com/fan-wenjie/[A-Za-z0-9_.-]+"), UPSTREAM_URL),
    (re.compile(r"https://gitee\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"), "<omitted>"),
    (re.compile(r"[A-Za-z0-9_.+-]+@(?:mail\.)?ustc\.edu\.cn"), "<omitted>"),
    (re.compile(r"[A-Za-z0-9_.+-]+@yottalabs\.ai"), "<omitted>"),
    (re.compile(r"\bfan-wenjie\b|\bfanwj\b", re.I), "<author>"),
]
# What --verify refuses to ship. Kept separate from REDACTIONS: a rule that
# stops matching is a silent hole, and this is what catches it.
FORBIDDEN = re.compile(
    r"fan-?wenjie|fanwj|\bustc\b|yottalabs|/home/[A-Za-z0-9_.-]+|"
    r"github\.com/fan|gitee\.com|樊文杰",
    re.I,
)

TEXT_EXT = {".py", ".md", ".txt", ".sh", ".json", ".jsonl", ".tex", ".yaml", ".yml",
            ".cfg", ".ini", ".toml", ".csv", ".log", ".patch"}


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def redact(text):
    for pat, rep in REDACTIONS:
        text = pat.sub(rep, text)
    return text


def _excluded(path_from_repo):
    return any(path_from_repo == e or path_from_repo.startswith(e + os.sep)
               for e in EXCLUDE_FILES)


def copy_tree(src, dst, stats):
    """Copy, redacting text files on the way through."""
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs
                   if d not in EXCLUDE_GLOBS
                   and not _excluded(os.path.relpath(os.path.join(root, d), REPO))]
        for f in files:
            if f in EXCLUDE_GLOBS:
                continue
            s = os.path.join(root, f)
            if _excluded(os.path.relpath(s, REPO)):
                stats["excluded"] += 1
                continue
            d = os.path.join(dst, os.path.relpath(s, src))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            ext = os.path.splitext(f)[1].lower()
            if ext in TEXT_EXT:
                try:
                    raw = open(s, encoding="utf-8", errors="replace").read()
                except OSError:
                    shutil.copy2(s, d)
                    continue
                new = redact(raw)
                if new != raw:
                    stats["redacted"] += 1
                open(d, "w", encoding="utf-8").write(new)
            else:
                shutil.copy2(s, d)
            stats["files"] += 1


def build_patch(out_dir):
    """The VestigeKV change as a single diff against the pinned upstream commit.

    `git diff` rather than `git format-patch`: the latter writes From/author
    headers, which is precisely what must not be in the archive."""
    eng = os.path.join(REPO, "engine")
    os.makedirs(os.path.join(out_dir, "engine"), exist_ok=True)
    patch = run(["git", "diff", "--binary", UPSTREAM_BASE, ENGINE_BRANCH,
                 "--", ".", ":(exclude).claude"], cwd=eng)
    patch = redact(patch)
    dest = os.path.join(out_dir, "engine", "vestigekv.patch")
    open(dest, "w", encoding="utf-8").write(patch)
    n_files = patch.count("\ndiff --git ") + patch.startswith("diff --git ")
    return dest, n_files, len(patch)


def verify(out_dir):
    """Refuse to ship on any surviving identifier, PDF text included."""
    hits = []
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, out_dir)
            ext = os.path.splitext(f)[1].lower()
            if ext == ".pdf":
                try:
                    import warnings
                    warnings.filterwarnings("ignore")
                    from pypdf import PdfReader
                    text = " ".join((pg.extract_text() or "") for pg in PdfReader(p).pages)
                except Exception as e:                      # noqa: BLE001
                    hits.append((rel, f"PDF unreadable, cannot clear it: {e}"))
                    continue
            elif ext in TEXT_EXT or ext == "":
                try:
                    text = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
            else:
                continue
            for m in set(FORBIDDEN.findall(text)):
                hits.append((rel, m))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="staging directory for the archive")
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip")
    ap.add_argument("--no-verify", action="store_true", help="skip the identity scan (don't)")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    stats = {"files": 0, "redacted": 0, "excluded": 0}
    for name in sorted(INCLUDE):
        src = os.path.join(REPO, name)
        if not os.path.exists(src):
            print(f"  ! missing, skipped: {name}", file=sys.stderr)
            continue
        dst = os.path.join(out, name)
        if os.path.isdir(src):
            copy_tree(src, dst, stats)
        else:
            open(dst, "w", encoding="utf-8").write(
                redact(open(src, encoding="utf-8", errors="replace").read()))
            stats["files"] += 1
        print(f"  + {name}")

    # Explicit source->dest mapping: two of these are called README.md and a
    # basename lookup would give the archive root's copy to both.
    EXTRAS = {
        "README.md": "README.md",
        "engine/setup_engine.sh": "setup_engine.sh",
        "engine/README.md": "engine_README.md",
    }
    for extra, srcname in EXTRAS.items():
        src = os.path.join(REPO, "tools", "supplement", srcname)
        if os.path.exists(src):
            d = os.path.join(out, extra)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(src, d)
            if extra.endswith(".sh"):
                os.chmod(d, 0o755)
            stats["files"] += 1
            print(f"  + {extra}")

    dest, n_files, n_bytes = build_patch(out)
    print(f"  + engine/vestigekv.patch  ({n_files} files, {n_bytes/1024:.0f} KB, "
          f"applies to sglang {UPSTREAM_BASE[:10]})")

    if not args.no_verify:
        hits = verify(out)
        if hits:
            print(f"\nIDENTITY SCAN FAILED -- {len(hits)} hit(s), nothing shipped:", file=sys.stderr)
            for rel, m in hits[:40]:
                print(f"  {rel}: {m!r}", file=sys.stderr)
            return 2
        print("  identity scan: clean")

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(out) for f in fs)
    print(f"\n{stats['files']} files, {stats['redacted']} redacted, "
          f"{stats['excluded']} excluded, {total/2**20:.1f} MB")

    if args.zip:
        zpath = out.rstrip("/") + ".zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for r, _, fs in os.walk(out):
                for f in sorted(fs):
                    p = os.path.join(r, f)
                    z.write(p, os.path.join(os.path.basename(out), os.path.relpath(p, out)))
        print(f"{zpath}  {os.path.getsize(zpath)/2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
