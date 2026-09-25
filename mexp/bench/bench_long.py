"""bench_one_batch_server driver: raises the 600s client timeout to 3600s
(the quadratic 512k prefill takes ~13 min; the default timeout turns that
point into a null). Does not modify the sglang tree."""
import sys

import sglang.benchmark.endpoint as _e
import sglang.benchmark.one_batch_server as _m

_m.DEFAULT_TIMEOUT = _e.DEFAULT_TIMEOUT = 3600
if hasattr(_m, "requests"):
    pass  # the timeout is read as a module global; that is sufficient
from sglang.benchmark.one_batch_server import cli_main

sys.exit(cli_main())
