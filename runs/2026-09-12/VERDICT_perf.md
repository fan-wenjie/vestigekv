# VERDICT — 性能线 P1/P2（2026-09-12）

配置：单节点 TP=2（2×RTX PRO 6000 96GB），`NCCL_P2P_DISABLE=1` +
`--disable-custom-all-reduce`，radix cache 开（生产一致），vk 弧
`SGLANG_VESTIGEKV_ACTIVATION_MIN_TOKENS=0`。客户端：sglang 官方
`bench_serving`；每 token 权威数据：服务器 decode 日志
（gen throughput，ERRATA #19：客户端流式路径惩罚更快的一臂）。

## P1 — bs=1 持续解码 4k→512k（双模型双臂，exit=0 ×4）

服务器侧中位 tok/s（4k 桶中位；Base 与 Instruct 逐桶一致到小数位）：

| 上下文桶 | vestigekv | dense | 比值 |
|---|---|---|---|
| 4k-32k | 247.7 | 249.2 | 0.994 |
| 32k-64k | 242.2 | 235.3 | 1.029 |
| 64k-128k | 239.5 | 219.0 | 1.094 |
| 128k-256k | 231.8 | 193.8 | 1.196 |
| 256k-384k | 223.4 | 168.8 | 1.323 |
| 384k-524k | 212.8 | 148.7 | 1.431 |

- 交叉点 ~28k（20k-28k 桶 0.999 → 28k-36k 桶 1.005）。
- 冻结参考点：@272k = 1.29×（冻结 bar：>1），@496k = 1.49×（bar：>1）→ **PASS**。
- 斜率：vk ~1.6 ns/token/step vs dense ~6.3 ns，比值 0.25 ≈ 字节账
  ρ+k_s/k = 0.26；固定部分 C0+T_net ≈ 3.9 ms（dense 曲线截距）。
- 4k 桶 vk 慢 0.6%（固定成本口径成立）。
- 数据：`results/latency_stream_serverlog_{vestigekv,dense}[_instruct].log`
  （各 13004 行，每 40 步一行）；客户端 jsonl 同目录保留但仅作参考。
- 图：`results/fig_latency_curve.png`（4k 桶中位，server-log splice，
  `mexp/bench/plot_curve_jsonl.py`）。

## P2 — 64k prefill + 4k decode 吞吐扫档（bs 1/2/4/8/12/16/24/32）

output tok/s：

| bs | Base vk | Base dense | Instruct vk | Instruct dense |
|---|---|---|---|---|
| 1 | 163 | 158 | 164 | 158 |
| 2 | 256 | 241 | 259 | 243 |
| 4 | 352 | 331 | 365 | 336 |
| 8 | 416 | 381 | 438 | 388 |
| 12 | 545 | 485 | 578 | 498 |
| 16 | 643 | 565 | 704 | 585 |
| 24 | 640 | 561 | 697 | 577 |
| 32 | 354 | 330 | 367 | 332 |

- vk 每一档领先，bs=12：Base +12.4%（冻结 bar：>0，参考 +8%）→ **PASS**；
  Instruct +16.1%；峰值优势 ~1.14–1.21×。
- bs=32 双臂同步回落：分块 prefill 占窗伪影，非方法效应（图注已注明）。
- 扫档 24/32 为冻结后用户决定的扩展，已记录于 README P2 段。
- 数据：`results/throughput_64k+4k_bs{...}_{vestigekv,dense}[_instruct].jsonl`；
  图：`results/fig_throughput.png`。

## 论文宏刷新

`\serveCrossover` ~28k、`\serveSpeedTwoFiveSix` 1.28×、`\serveSpeedMax`
1.49×、`\tputGainTwelve` 12.4%、`\srvFixed` 3.9（替换旧 C0/T_net 占位）、
`\srvAdenseTk` 0.41；sec:cost 的 memory-time 交叉 60k→128k、
496k 比值 0.82→0.77；注意力占比 7%→10%。
