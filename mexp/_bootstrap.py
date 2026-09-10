"""Path bootstrap: import sglang from a source tree directly (no pip install).

The experiment code depends ONLY on sglang (submodule ./sglang, branch
vestigekv). The former mini-sglang reference engine is retired; mexp scripts
that still `from minisgl...` are frozen archival records of published gates --
they raise ImportError here by design rather than silently resolving to a
stale copy. Experiment scripts do `import _bootstrap` first."""
import os
import sys

_CANDIDATES = (
    os.path.join(os.path.dirname(__file__), "..", "engine", "python"),  # submodule form
    "/home/user/fft/sglang/python",                                     # sibling form
)
for _p in _CANDIDATES:
    _p = os.path.abspath(_p)
    if os.path.isdir(os.path.join(_p, "sglang")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break
else:
    raise ImportError("sglang source tree not found; init the submodule")
sys.path.insert(0, "/home/user/fft/nope_kv")
