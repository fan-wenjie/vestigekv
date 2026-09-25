# 论文最初方案 vs 代码实现 — 差异清单

本文档对照 **论文**（`/home/user/vestigekv/paper/main.tex`，ICLR 2027 draft v8，
2026-09-09；含 `sec_eng_serving.tex`）与 **代码实现**
（`engine/` commit `e44b914a7`，branch `vestigekv`），逐条列出"论文写的最初方案"
在实现过程中被修改的地方。

> **状态更新（2026-09-12）**：论文已按本文档逐条修改对齐代码（以代码为基准），
> 匿名版与 named preprint 版均 pdflatex 编译通过（19/18 页，零警告）。
> 本文档保留为**修改记录/变更日志**，记录每条差异的原始表述与修改依据。

背景说明：

- `sec_eng_serving.tex`（§3.3 "The attention implementation on a production stack"）是
  serving 移植完成后补写的小节，其口径已与新实现对齐；**差异主要集中在 main.tex 的
  §3.2（recall tier）、§4（recommended configuration）与 Appendix D（Deployment
  specification）**——这些章节保留着 harness 时代的最初方案。
- 论文内部本身存在前后矛盾（§4 已改口、附录 D 未改），下文逐条注明。
- 本文档与 `DIFFS_zh.md`（白皮书 v1→v2 修正记录）互补：DIFFS 记录白皮书怎么改，
  本文档记录论文方案与最终实现差在哪。
- 代码行号基准：`engine/python/sglang/srt/layers/attention/` 下的
  `vestigekv_mla_backend.py`（简称 backend）与 `vestigekv/` 包
  （`defaults.py` 简称 D、`recall_tier.py`、`eviction.py` 等）。

---

## 差异 1：索引构建时机与校准查询来源（最大的一处方案变更）

- **论文最初方案**：§3.2（main.tex:361）写 sketch 基 "$V_r$ from the context's own
  **prefix queries**"；附录 D item 3（main.tex:1142-1145）写 "Compression event
  (**prefill end**): tier-2 index build … $z$ and the entropy gate self-calibrate
  from the prefix"。即：prefill 结束时用 prefill 查询一次性建好校准索引。
- **代码事实**：prefill 查询在服务栈里**不可用**——它们未被 absorb（形状
  `[·, H, 192]`，吸收成 576 维需要模型权重 $W_{kc}$，backend 拿不到），且 chunked
  prefill 下"哪一块是最后一块"从 attention backend 无法判定（backend:966-970 注释）。
  实现改为**两阶段**（backend:1486-1538）：
  1. 首个 decode 步**同步**建 provisional 索引：V = 单位阵前 r 行（未校准基）、
     熵门全开（θ_g = −∞）、zp 钳到 Z_MAX = 8 —— 宁多取、绝不漏；
  2. 收集 8→64 步真实 decode 查询（`N_CAL_START`/`N_CAL_MAX`，D:82-92）后，
     在**侧流异步**建 calibrated 索引并原子安装，带请求身份校验防止 radix-cache
     slot 复用后装错索引。
- **曾走过并被否定的路线**：用 cache 行自身作代理查询——恒以自身为 argmax、且恒落
  在 attended tail 内，导致 n_hard = 0、fire 率膨胀约 60×，废弃。
- **论文已部分改口**：`sec_eng_serving.tex:32-35` 已写为实现口径
  （"Calibration fits on the request's first decode queries with a provisional
  index serving from step one; the O(S) calibrated build runs on a side stream"）。
  §3.2 与附录 D item 3 未同步。

## 差异 2：低通带宽 κ —— 从"随长度变化"到固定 16

- **论文最初方案**：推荐配置表（main.tex:951）写 "κ = 16 ($L\le16$k); 64
  ($L\ge32$k)，tracks context length"。
- **代码事实**：`LOWPASS_KAPPA = 16` 固定（D:47-55）。注释记录了测量结论：
  query-grounded 准则（σ_κ top-(ρT) 集捕获的注意力质量）恰好峰值在 16，且
  κ∈[4,64] 上平坦到 ±0.3%，数据驱动选择无可赢；唯一便宜的 query-free 代理
  （σ 的峰度）会错误地选 32。故固定。
- **注意**：main.tex:557 的 §4 正文已把 "κ=16" 列为 structural constant，与
  同文配置表（main.tex:951）自相矛盾——正文是新口径，配置表是最初方案残留。

## 差异 3：fetch 上限 —— 附录写 top-j=16，实现默认 uncapped

- **论文最初方案**：附录 D item 4（main.tex:1146-1148）写 "fetch the
  **top-$j$=16** fired rows into this step" 并标 [M]。
- **代码事实**：默认 `topj = -1`（UNCAPPED，取全部 fired 集；backend:78-80、114）。
  只有显式设 `SGLANG_VESTIGEKV_TOPJ` 才启用上限。`FETCH_WIDTH_UNCAPPED = 4096`
  （D:101-104）只是固定地址缓冲的宽度（高于实测最坏 fire 3983），不是上限。
- **论文已部分改口**：§4（main.tex:575-577）与配置表（main.tex:957-959）均写
  "uncapped by default … topj=16 is an optional worst-case-latency bound"——
  以§4口径为准，附录 D item 4 过期。

## 差异 4：熵门 —— 机制完整，但在 serving 实践中自我关闭

- **论文最初方案**：§3.2/§4 把熵门作为活跃的预算分配组件
  （α = (1−τ)/2 的假关闭预算，main.tex:380-384、548）。
- **代码事实**：机制完整实现（recall_tier.py:355-406），但校准后若拟合阈值会放行
  超过 60% 的校准查询，门就**自我禁用**（θ_g = −∞，
  `GATE_SELF_DISABLE_FRACTION`，D:76-80；recall_tier.py:405-406）——"一个在丢弃
  什么的阈值才配存在"。serving 移植路径上实测即落入自关分支（校准后门放行
  >60% 查询）。配置表（main.tex:961）其实已写 "auto-off if cal. fire rate >60%"，
  与代码一致；差异在于正文把它当作常态组件而非一个会自我裁掉的优化。

## 差异 5：per-layer index cascade —— 论文提到，发布代码中不存在

- **论文最初方案**：§4（main.tex:561-563）与附录 D item 5（main.tex:1149-1152）：
  可选的逐层级联（stage-1 sidecar shortlist），保持 recovery 不变、scan 降到
  ×0.65（8k）/×0.79（32k），uniform 变体被否且"must not ship"。标 [M]。
- **代码事实**：`vestigekv/` 包与 backend 中 grep 不到 cascade/shortlist 的任何
  实现——这是 harness 时代的测量项，**未移植**到发布的 serving backend。

## 差异 6：tier-1 规模调度 —— constant-m 改为 proportional-m（论文内部矛盾）

- **论文最初方案**：§4（main.tex:565-566）与附录 D "Placement and knobs"
  （main.tex:1155-1156）写 "The ratio floor for a **constant-$m$** schedule is
  1/32 — the only ratio measured flat in length"。
- **代码事实**：部署形态是 **proportional**：每次 close 重平衡时
  `m = max(1, round(ρ·closed))`（backend:1461；另见 backend:1994 注释
  "m=round(rho*seq_len)"），行在两层间双向移动。`select_kept` 留有 `m_fixed`
  参数，但那不是部署形态。
- **论文已部分改口**：`sec_eng_serving.tex:82-83` 自己写的就是 "The tier-1 set is
  **proportional** ($\rho T$ plus the tail; the only schedule measured quality-flat
  in length)"——同一篇论文两处表述矛盾，以 serving 小节与代码为准。

## 差异 7：激活阈值 L* —— spec 有，代码最初无显式开关，现已实现

- **论文最初方案**：附录 D item 1：低于激活阈值 L* 时 partition 完全关闭——
  无索引、无 compaction、与 stock MLA 路径逐字节一致、零附加成本。[D]
- **代码事实（初版）**：**没有显式的 L\***。短上下文下行为等价来自块粒度本身：
  `closed = ⌊seq_len/4096⌋·4096`，seq_len < 4096 时 closed = 0，archive 为空、
  kept 即全部行。但每步融合 kernel 与索引构建**照常运行**（对空 archive 的 scan
  很便宜但非零）——"零附加成本"是设计目标（[D]），不是实现事实。
- **代码事实（现版，2026-09-12 起）**：**已实现显式 L\* =
  ACTIVATION_MIN_TOKENS = 32768**（8×CLOSE_BLOCK，defaults.py）。低于阈值时
  请求完全走 dense：不闭合块、不归档、不建索引、不收集校准查询，kept = 全部行
  （与 stock 逐字节一致）；召回 kernel 侧增加空归档早退（prologue scores/merge、
  scan、compact_write 对 a_len==0 的 pair 零迭代退出），且 scan/compact 改为
  grid-stride（每 pair 上限 SCAN_GRID_CAP=128 个 program）。动机是实测：召回管线
  固定成本 ~114 µs/step（torch profiler），而 32k 以下 dense 注意力比
  kept+scan+fetch 更便宜（注意力单独交叉点 ~32k，端到端 ~28k，2026-09-12
服务端日志口径）。跨过阈值的
  那一步 close 循环一次补齐所有整块——块级 σ 一经计算不可变、全局 top-m 重平衡
  与闭合次序可交换，kept 集与从头压缩完全一致。修复后每步召回管线
  ~114 µs → ~18 µs（空归档早退 + grid-stride），8k 短上下文与 dense 基线持平。

## 差异 8：Kernel mapping —— 从"不写新 kernel"到七个自研融合 Triton kernel

- **论文最初方案**：附录 D "Kernel mapping"（main.tex:1171-1175）："No new
  kernels … the index scan is **one GEMV per layer**; fetched rows merge as one
  extra split-KV partition"；熵门"consume the LSE that kernel already emits"。[D]
- **代码事实**：召回路径实现为 **7 个自研融合 Triton kernel**，全部烘焙进 decode
  CUDA graph：prologue 2（split-NK 在线 softmax + merge/熵门/qsk·qres 勾股投影）→
  scan 1 → compact 2（counts 前缀和 + 保序写入 fetch_buf）→ CSR pack 2。
  熵门与逐头 max1 由 fused prologue **自己**对 kept 行算在线 softmax 得到，
  并不消费 stock attention kernel 的 LSE。
- **论文已部分改口**：`sec_eng_serving.tex:24-37` 已更新为七 kernel 口径；
  其中 "No kernel is written" 仅指**未改写 stock MLA-decode attention kernel
  本身**（wrapper 只改它读哪些行），不再指整个召回路径。

## 差异 9（措辞级）："recent window" 的语义

- **论文**：§3.1（main.tex:331）写 "plus 4 sinks and **the recent window**"。
- **代码事实**：始终出席的尾部是**未闭合余量**（< CLOSE_BLOCK=4096 行，含上次
  close 以来 decode 出的全部 token），没有固定宽度的 recent window；
  `RECENT_WINDOW=256` 仅剩 debug 行不变量检查器的尺度用途（backend:1052 注释）。
- 配置表（main.tex:952）写 "first 4 / one block" 反而是对的；§3.1 的
  "recent window" 措辞过期。

## 差异 10（细节澄清）：fire 判定的粒度

- **论文**：§3.2（main.tex:373）写 "fetch the top-$j$ rows whose score exceeds
  **the tier-1 maximum**"（单数），图 2 同（"top-$j$ > $s^{*}$"）。
- **代码事实**：判定是**逐头**的——每个头有自己的 kept 最大值 max1
  （fused_prologue.py:38 `max1g_ptr [P, H]`），任一头上的 certified score 超过
  该头 max1 即 fire，最终 fetch 集是各头 fired 集的**并集**。论文公式层面
  $s_h(t,u)$ 本来就带头指标，此处只是正文表述把逐头竞争略写成了单一阈值。

---

## 附记：本次复现相对论文测量协议的偏差（非算法差异）

- **并行方式**：论文 serving 数字为两卡 **pipeline parallel**
  （sec_eng_serving.tex:13）；本次复现跑 **TP=2**（`--disable-custom-all-reduce`）。
  backend 每个 rank 完全本地、无集合通信，两种并行方式均支持。
- **质量数据集**：论文质量表为 gsm8k 64-shot（n=800，main.tex:837）；本次复现因
  gsm8k 数据集问题改用 **gsm8k-platinum 全部 1209 题**（见仓库根 ERRATA 记录）。
- **Mamba 缓存约束**：复现时须显式设 `--max-running-requests`（Metric 1 用 4，
  Metric 2 用 32），否则 Mamba 缓存按 835 请求预分配 ~16.33 GB 会在约 478k
  上下文处 OOM。

## 已核对一致、不属于差异的部分

| 项目 | 论文位置 | 代码位置 |
|---|---|---|
| σ：逐 4096 块定窗 rFFT 低通残差（非整前缀） | §3.1，main.tex:316-330 | eviction.py:78-116 |
| live rebalance：每 4096 个 decode token 闭块、全局 top-m 重平衡 | §3.1，main.tex:336-342 | backend:1439-1483 |
| 七融合 kernel 烘焙进 decode CUDA 图 | sec_eng_serving.tex:24-37 | batched_step.py / fused_prologue.py 等 |
| 16-bit 索引 ~260 B/行，读字节账 1152→~294 B/token | §2.4 Eq.(reads)，main.tex:255-271 | recall_tier.py（条目布局） |
| 闭式 conformal z：k=⌈(n+1)τ_s⌉、min_hard=18、τ=0.90、Z_MAX=8、8→64 窗口 | §4，main.tex:547-554 | D:63-92、166-201；recall_tier.py:409-456 |
| 架构前提检查（拒绝非 NoPE-MLA、page_size≠1、投机解码），失败即 raise | sec_eng_serving.tex:16-22 | backend（初始化检查） |
| 异步合同：close 工作可离关键路径，滞后只保守地放大 attended set | 附录 D，main.tex:1163-1169 | backend close 路径 |
| 部署不变式"只分区、不删行"，kill-switch 可逐字节回到 stock MLA | §3.1、图 1 | backend（wrapper 架构） |
