# Running something on this line

## The served configuration

`mexp/glm53/baseline.sh` + `common.sh`. The DSA baseline and every vk arm are the
SAME arm (`arm: baseline`) on the SAME tree -- only env vars differ. That is what
makes the comparison clean, and it is worth preserving.

    --model-path nvidia/GLM-5.3-Flash-NVFP4 --quantization modelopt_fp4 --tp-size 2
    --kv-cache-dtype bfloat16 --moe-runner-backend flashinfer_cutlass
    --mem-fraction-static 0.955 --context-length 73728 --max-running-requests 4
    --max-mamba-cache-size 4 --chunked-prefill-size 1024 --language-model-only
    --disable-radix-cache --disable-custom-all-reduce --sampling-backend pytorch
    --random-seed 0 --reasoning-parser glm45
    --dsa-prefill-backend triton --dsa-decode-backend triton     <- SM120 needs these

## The selector's knobs

    SGLANG_DSA_VK_SELECT          0 = DSA's own radix top-k (the reference arm)
                                  1 = the reference-set threshold (the vk arm)
    SGLANG_DSA_VK_SELECT_Q        0.0 only. Any q>0 ratchets the threshold shut:
                                  measured 512 -> 130 -> 35 -> 9 groups at q=0.75
                                  with recall 1.00 -> 0.02, served multikey 0.72.
    SGLANG_DSA_VK_SELECT_RECENT_W 0. W=16 is refuted; see state.md.
    SGLANG_DSA_VK_SELECT_TELEMETRY 1 logs VKRECENCY per layer. DIAGNOSTIC ONLY --
                                  it costs an extra compare pass plus an RxG rank
                                  reduction, and the readout only fires when the
                                  stream is NOT capturing, so decode CUDA graphs
                                  must be disabled with
                                  --cuda-graph-backend-decode=disabled or it
                                  measures extend steps only.
    SGLANG_DSA_INDEX_TOPK         experiment knob; it is BOTH the selection budget
                                  and the attention row count. The installed
                                  sgl_kernel asserts topk==2048 at all three entry
                                  points, so this axis is closed.

## Queueing a job

1. **Copy an existing row** from `mexp/glm53/queue.jsonl` -- each client's args fail
   differently and two of them fail silently.
2. Append, then **verify the runner's own view** (it re-reads the file the instant a
   job ends; a late head insertion is silently skipped):

        python3 - <<'P'
        import sys; sys.path.insert(0,"mexp/glm53"); sys.path.insert(0,"mexp")
        import queue_runner as qr, naming
        epoch = naming.tree_epoch(qr.ROOT)
        done = {r["id"] for r in qr.read_jsonl(qr.STATE)
                if r.get("status") in ("done","failed") and r.get("end","") >= epoch}
        print([j["id"] for j in qr.read_jsonl(qr.QUEUE)
               if not j.get("skip") and j["id"] not in done])
        P

3. **Register it in `README.md`** with the acceptance stated BEFORE the data, and
   the verbatim JSON row. The audit reads the register.
4. Launch, and only when nothing else is alive:

        nohup .../python mexp/glm53/queue_runner.py > results/glm53/runner_X.log 2>&1 &

## Clients

`ruler` (lm-eval v1 group; `tasks`, `lengths`, `n`), `stream` (decode-only latency --
the ONLY source of a cost claim), `copy`, `profile`, `replay`, `longbench1`, `gsm8k`,
`phase` (`phase_needle.py`; token-exact needle placement via input_ids, with
`distractors` and `phases`).

## Always ride a canary

After any selector change, include `niah_multikey_3` in the job. It reads 1.000 on
the vk arm in three independent runs, so a drop is unambiguous. Both silent failures
on this line -- the trash slot, and W=16's `+inf` latch -- reproduced every byte,
fetch and ITL counter with `errs=0` and no traceback. The quality draw was the only
detector.

## Verifying a number before quoting it

- `python mexp/exp/check_builds.py` (in `/home/user/vestigekv`) before quoting any
  new arm -- the async build worker swallows exceptions and the provisional index
  serves silently.
- Cost claims only from decode-only `stream` jobs, quoting median AND p99 ITL.
  Never MRCR wall time, never ruler wall time.
- A set env var is not a consulted one. Trace it to the value the kernel receives:
  read `/proc/<pid>/environ` of the server, and for a JIT kernel check
  `/proc/<pid>/maps` of the scheduler ranks -- both must map the SAME build hash,
  or the ranks are on different kernel source (this voided `c521vkpf`).
