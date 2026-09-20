# Quarantine manifest: streaming-client ITL records removed 2026-09-20

These files are the per-token inter-token-latency records written by
sglang's streaming benchmark client on long continuous-decode runs. They are
deleted, not archived, because their metric is invalid by this project's own
diagnosis and because shipping them actively misleads.

WHY THEY ARE WRONG (runs/2026-09-11/ERRATA.md, item 19). With
stream_interval=1 the server emits one detokenize + SSE event per token, and
the cost of that path grows with stream length. The faster arm generates
faster, accumulates a deeper send backlog, and therefore reports a LONGER
client-side latency: the metric penalises the faster side. On the P1 v2 pair
the server finished generating at 2297 s while the client was still draining
at 5308 s. The bottleneck was isolated to the server's main process
(tokenizer manager + SSE), not the model, the backend or the client.

WHAT THAT DID TO THE NUMBERS. Reduced, these files give a dense/VestigeKV
window-median ratio of 0.066 at 508k -- VestigeKV fifteen times slower --
where the server-side log for the same run gives 1.431 for the 384k-524k
bucket. A reader who unzipped the archive would compute the first.

WHAT REPLACES THEM. The server-side decode log, which this paper already
declares the per-token authority, produced by mexp/exp/latency-pair-512k.sh
with both arms on one tree and the log retained.

No macro in the paper cited any file below; deleting them changes no number.

| file | bytes | sha256 |
|---|---|---|
| `results/kimi/latency_stream_4k-126976_vestigekv_stats.jsonl` | 91223 | `bb9ea85654317a34ab68aeace57425e0` |
| `results/kimi/latency_stream_4k-253952_vestigekv_fence-mk256-stream-256k.jsonl` | 97982 | `a1e035a5e2bd0b4a82a267ce3e1a9bf8` |
| `results/kimi/latency_stream_4k-253952_vestigekv_stats_fence-mk256-stats-256k.jsonl` | 97717 | `f4e92880ae1d2603f76df1400920bb75` |
| `results/kimi/latency_stream_4k-258048_baseline.jsonl` | 194901 | `3d98f85ca97b40b95b101540d6c0eedd` |
| `results/kimi/latency_stream_4k-258048_vestigekv.jsonl` | 486938 | `7c534c64c92c5635227a3eb8e389b000` |
| `results/kimi/latency_stream_4k-258048_vestigekv_prod-stream-vestigekv-256k-nofence.jsonl` | 97254 | `56eab59012c43782a6c5e3ca66186afc` |
| `results/kimi/latency_stream_4k-258048_vestigekv_prod-stream-vestigekv-256k-origin.jsonl` | 96219 | `ee0053119b2665e82b524c60ee64c333` |
| `results/kimi/latency_stream_4k-258048_vestigekv_prod-stream-vestigekv-256k-perf-nofence.jsonl` | 97837 | `2a5f1aefe84f14ad6156b158519f3265` |
| `results/kimi/latency_stream_4k-258048_vestigekv_stats.jsonl` | 193696 | `49d64e1dafb970ba2c415b264ae46481` |
| `results/kimi/latency_stream_4k-258048_vestigekv_stats_pc-A1-stream256k-stats.jsonl` | 97252 | `5e168ad620b48d9accaf6fce7d7a2c0b` |
| `results/kimi/latency_stream_4k-258048_vestigekv_stats_stats-vestigekv-stream-256k-m2.jsonl` | 97197 | `2be8ff4ca84b8d1e9a4456b68b691fae` |
| `results/kimi/latency_stream_4k-258048_vestigekv_stats_stats-vestigekv-stream-256k-perf-rebuild.jsonl` | 97420 | `7c379b1f30cfe12b3343a2aa1b8516c2` |
| `results/kimi/latency_stream_4k-258048_vestigekv_stats_td-stats-256k.jsonl` | 97487 | `5fbc055a6e78b112aaa841f09193ef5f` |
| `results/kimi/latency_stream_4k-258048_vestigekv_stream-vestigekv-256k-m2.jsonl` | 97272 | `528b51d0c9099b5cad21292451bb3d1e` |
| `results/kimi/latency_stream_4k-258048_vestigekv_stream-vestigekv-256k-origin.jsonl` | 96408 | `0cbcf866ccdd3e803428ae875dc80fb9` |
| `results/kimi/latency_stream_4k-258048_vestigekv_stream-vestigekv-256k-perf-rebuild.jsonl` | 97465 | `9af97a62ec539cc872e2a1e93eeb7690` |
| `results/kimi/latency_stream_4k-258048_vestigekv_stream-vestigekv-256k-perf-splits.jsonl` | 97428 | `ca7bb4b869d0e67bf0546175d9f4ec3f` |
| `results/kimi/latency_stream_4k-258048_vestigekv_stream-vestigekv-256k-perf.jsonl` | 97360 | `bb3e7d5bcbf1093408d995065f0e85e9` |
| `results/kimi/latency_stream_4k-258048_vestigekv_td-stream-256k-affine.jsonl` | 98048 | `099785228f104a7884fcbdc753ad1870` |
| `results/kimi/latency_stream_4k-258048_vestigekv_td-stream-256k-nodense.jsonl` | 97914 | `5cd3fa76347b8d0aa17caaccba838b86` |
| `results/kimi/latency_stream_4k-258048_vestigekv_td-stream-256k-nospill.jsonl` | 97523 | `0750a58723adaf2d6da57f42eeed4cab` |
| `results/kimi/latency_stream_4k-258048_vestigekv_td-stream-256k.jsonl` | 97887 | `c5088392c18df9f0ae80d2a0c9ae264c` |
| `results/kimi/latency_stream_4k-520192_vestigekv.jsonl` | 107506 | `a7d014dc077303841535d0154e825f0f` |
| `results/kimi/latency_stream_64k-14-x130_vestigekv_stats.jsonl` | 178131 | `6e0004b3d554bc9ab8acd4cf5dfb98b4` |
| `results/kimi/latency_stream_64k-4096_vestigekv_stats.jsonl` | 420317 | `9e059eb17ae89aa45ea360eeb3093002` |
| `results/kimi/latency_stream_64k-64-x2_vestigekv.jsonl` | 41515 | `54996503900f61c880d899a5ac784097` |
| `results/kimi/latency_stream_64k-64-x30_vestigekv_stats_current-stats-64k-eager.jsonl` | 176806 | `ae653d35722293cb1537930eb4acb784` |
| `results/kimi/latency_stream_64k-64-x30_vestigekv_stats_origin-stats-64k-eager.jsonl` | 174069 | `c77f25233604c6b24dc1b8fb6e2301d6` |
| `results/latency_stream_4k-512k_dense.jsonl` | 427239 | `bc55ed47882d6ba06f09e8fb77515021` |
| `results/latency_stream_4k-512k_dense_instruct.jsonl` | 213174 | `7500695429d9a9b5a50fcfa10e45a28b` |
| `results/latency_stream_4k-512k_vestigekv.jsonl` | 427658 | `11b494d04d921e15a5dacd19a0f73053` |
| `results/latency_stream_4k-512k_vestigekv_instruct.jsonl` | 213355 | `23883067afc2c50f689f1eb8c3424c35` |

Also removed from results/results.zip on rebuild: 35 entries matching
`latency_stream_*` (5.6 MB of 31.6 MB uncompressed).

