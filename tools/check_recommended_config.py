"""Consistency check: config/recommended.json is the authority, not a wish.

Fails (exit 1) when a value the JSON claims diverges from what the code /
launch script actually carries. Checks only what it can resolve; a check that
cannot find its subject FAILS LOUDLY instead of passing on emptiness.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.load(open(ROOT / "config" / "recommended.json"))
SGL = ROOT.parent / "sglang" / "python" / "sglang" / "srt" / "layers" / "attention"

failures, checked = [], 0


def check(name, actual, expected):
    global checked
    checked += 1
    if str(actual) != str(expected):
        failures.append(f"{name}: json says {expected!r}, source has {actual!r}")


# algorithm constants now live in one module (vestigekv/defaults.py); the backend
# only references them. Read them there, and refuse to pass if the module or any
# named constant is missing rather than silently checking nothing.
src = (SGL / "vestigekv_mla_backend.py").read_text()
dflt_path = SGL / "vestigekv" / "defaults.py"
if not dflt_path.exists():
    sys.exit(f"ABORT: {dflt_path} missing -- checker is checking nothing")
dflt = dflt_path.read_text()


def const(name, pattern=r"([-\d./ ]+)"):
    m = re.search(rf"^{name} = {pattern}$", dflt, re.M)
    if not m:
        sys.exit(f"ABORT: {name} not found in defaults.py -- checking nothing")
    return eval(m.group(1))


check("recent_window", const("RECENT_WINDOW"), CFG["recent_window"]["value"])
check("rho", const("RHO"), CFG["rho"]["value"])
check("sinks", const("SINKS"), CFG["sinks"]["value"])
check("index_rank", const("INDEX_RANK"), CFG["index_rank"]["value"])
# the backend must take these FROM the module, not re-declare its own copy
for name in ("RECENT_WINDOW", "SINKS", "RHO", "INDEX_RANK"):
    checked += 1
    if re.search(rf"^_{name} = ", src, re.M):
        failures.append(f"{name} re-declared in vestigekv_mla_backend.py")
# fool-proof invariant: the SERVING default for the cap must be uncapped (-1),
# NOT the recommended value -- the recommendation is typed by the engineer.
m = re.search(r"topj: int = (-?\d+),", src)
if not m:
    sys.exit("ABORT: topj default not found")
check("topj serving default (must stay -1/uncapped)", int(m.group(1)), -1)

# launch script carries no magic numbers for the operational keys
launch = (ROOT / "mexp" / "launch_vestige_server.sh").read_text()
for key in ("attention_backend", "sampling_backend", "mem_fraction_static",
            "context_length", "chunked_prefill_size", "pp_partition"):
    if f"cfg {key}" not in launch and "pp_partition" != key:
        failures.append(f"launch script does not source {key} from the json")
    checked += 1
for lit in ("0.88", "8192", "262144", "23,4"):
    # a literal appearing in the launch script means a magic number crept back
    if re.search(rf"(?<![\w.]){re.escape(lit)}(?![\w.])", launch):
        failures.append(f"magic literal {lit!r} found in launch script")
    checked += 1

if checked < 14:
    sys.exit(f"ABORT: only {checked} checks resolved -- subject drifted?")
if failures:
    print("\n".join("FAIL " + f for f in failures))
    sys.exit(1)
print(f"OK: {checked} checks, recommended.json consistent with sources")
