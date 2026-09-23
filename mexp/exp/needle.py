"""Needle at the head of a ~10k-token filler; the answer needs rows from the first closed block."""

import os
import json
import sys
import time
import urllib.request

SECRET = "The secret access code is 7-ZEBRA-4419."
FILLER = (
    "The committee reviewed the quarterly logistics report, noting that shipping volumes rose modestly "
    "while warehouse turnover slowed, and recommended a revised staffing plan for the northern depots. "
)
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 330
port = sys.argv[2] if len(sys.argv) > 2 else "30000"
body = (
    SECRET + "\n\n" + FILLER * reps
    + "\n\nQuestion: What is the secret access code stated at the very beginning of this document? Answer with the code only."
)
req = {"model": os.environ.get("NEEDLE_MODEL", "nvidia/GLM-5.3-Flash-NVFP4"), "messages": [{"role": "user", "content": body}], "max_tokens": 1024, "temperature": 0}  # 300 let the thinking eat the answer: a MISS with the code cut mid-string (2026-09-23)
t0 = time.time()
r = urllib.request.urlopen(
    urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}),
    timeout=3600,
)
out = json.load(r)
m = out["choices"][0]["message"]
content = m.get("content") or ""
print("usage:", out["usage"], f"wall={time.time() - t0:.0f}s")
print("reasoning tail:", (m.get("reasoning_content") or "")[-300:].replace("\n", " "))
print("content:", content[:200].replace("\n", " "))
print("NEEDLE_OK" if "7-ZEBRA-4419" in content or "7-ZEBRA-4419" in (m.get("reasoning_content") or "") else "NEEDLE_MISS")
