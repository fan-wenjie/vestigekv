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
ENG = ROOT / "engine" / "python" / "sglang" / "srt"
SGL = ENG / "layers" / "attention"

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
# The fetch cap: the json must state the flag's real default, not an aspiration.
# topj is gone from serving -- this used to assert topj == -1 against a source
# path that no longer existed, so it crashed instead of checking, which is how
# the json came to describe a knob the engine had removed.
args_src = (ENG / "arg_groups" / "fields" / "exec_.py").read_text()
m = re.search(r"vestigekv_recall_capacity:.*?\]\s*=\s*(\d+)", args_src, re.S)
if not m:
    sys.exit("ABORT: vestigekv_recall_capacity default not found")
check("recall_capacity default", int(m.group(1)), CFG["recall_capacity"]["value"])
checked += 1
# A KEY named topj, not the word: the recall_capacity entry explains on purpose
# that topj was removed, and a substring test flagged its own explanation.
if "SGLANG_VESTIGEKV_TOPJ" in (ENG / "environ.py").read_text() and "topj" in CFG:
    failures.append("recommended.json still has a topj key, which serving deprecated")
checked += 1

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
