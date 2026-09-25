# 白皮书（v1）与代码不一致清单 — v2 修正记录

本文档逐条列出 `whitepaper.tex` v1（2026-09-06，冻结于 commit `c3e2612`）与
当前代码（commit `e44b914a7`，branch `vestigekv`）之间的不一致，以及 v2 的修正。
基准：`engine/python/sglang/srt/layers/attention/vestigekv/` 与
`vestigekv_mla_backend.py` 的实际实现。

## 1. Tier-1 选择规则：固定 recent window 不存在（已修正）

- **v1 白皮书**：Eq.(2) 写作 keep = top-m(σ) ∪ sinks(前4行) ∪ 最近窗口 W=256；
  参数表列有 "W (recent) = 256，公认随意"。
- **代码事实**：`select_kept()`（eviction.py:41-55）只在**已闭合行**上做
  top-m ∪ sinks；始终出席的尾部是**未闭合余量**（< CLOSE_BLOCK=4096 行，
  含上次 close 以来 decode 出的全部 token），没有固定 256 窗口。
  `RECENT_WINDOW=256` 仅剩 debug 行不变量检查器的尺度用途
  （vestigekv_mla_backend.py:1052 注释明确说明）。
- **v2**：Eq.(2)、Algorithm 1、参数表均改为"未闭合尾部"语义。

## 2. σ 的变换窗口：逐 4096-block，而非整前缀（已修正）

- **v1 白皮书**：Eq.(1) 与 Algorithm 1 对整个前缀做长度 T 的 rFFT；
  数值注记写明"rFFT 长度是真实前缀长度 T"。
- **代码事实**：`blockwise_sigma`（eviction.py:78-116）对每个完整
  CLOSE_BLOCK=4096 块独立做定窗 rFFT。整前缀变换会让 κ=16 的截止周期随
  上下文漂移（T=4096 时 ≥256 token，T=512k 时 ≥32k token）；定块窗保证
  σ 尺度不变、逐块只算一次且不可变。该设计理由 v1 在参数表 κ 一行里其实
  提过，但正文公式与算法与之矛盾。
- **v2**：Eq.(1) 改为块内变换并说明理由；Algorithm 1 改为逐块循环。

## 3. σ 的实现路径：生产路径是融合 Triton kernel，不是 torch.fft（已修正）

- **v1 白皮书**："Implementation path: this stays framework torch"——
  声称选择路径有意保持 torch.fft/topk，Triton 化是"负优化"。
- **代码事实**：`SIGMA_FUSED=True`（eviction.py:24），CUDA 上走
  `sigma_fused`（sigma_fused.py）：rFFT→低通→irFFT 链以精确的正交投影
  闭式计算（σ_u = ‖R_u − c_uᵀ(CᵀR)‖，C 为块上固定三角基），并融合
  radix 直方图 + 精确 top-m（topm_from_hist）。实测 batched prefill
  快 28×，与 cuFFT 差 ~1e-7 rms（低于 bf16 输入量化三个数量级），
  top-m 100% 重合；torch 链保留为位一致 oracle（注册测试断言）。
- **v2**：该段整体重写为融合 kernel 描述，torch 链降为参照物。

## 4. 熵门分位数：3% → 5%（已修正）

- **v1 白皮书**：两处写"3%-quantile of entropy over hard queries"。
- **代码事实**：`gate_alpha(τ) = (1−τ)/2`（defaults.py:166-169），
  τ=0.90 时 α = **0.05**，即 5% 分位。
- **v2**：两处均改为 α=(1−τ)/2=5%。

## 5. 证书权重 z：11 档 ladder 已废弃，代码为闭式 conformal 分位数（已修正）

- **v1 白皮书**：§6.2 item 5 声称 ladder 已删除、由闭式分位数替代；但
  §6.3"Index build" item 7 仍描述从 {0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8}
  中选最小达标 z（ladder 残留，文档自相矛盾）；且 §6.3 item 5 引用实测
  "z ∈ {0, 0.5, 1, 1.5}"（ladder 时代的离散取值）。
- **代码事实**：`zp = min(kthvalue(z_req, k), Z_MAX)`，
  k = ⌈(n+1)·τ_s⌉（recall_tier.py:409-456），连续取值；
  校准在与服务 kernel 完全一致的量化后路径上进行（quantize-then-calibrate）。
- **v2**：item 7 改为闭式分位数并补充 quantize-then-calibrate 细节；
  实测取值描述改为 [0, Z_max] 连续。

## 6. 硬样本数门槛：">5" → 18（已修正）

- **v1 白皮书**：熵门校准"if >5 hard queries exist"。
- **代码事实**：`min_hard(τ=0.9) = ⌈τ_s/(1−τ_s)⌉ = 18`（defaults.py:191-201）；
  不足 18 个 hard 样本时门保持全开、zp 钳到 Z_MAX=8。
- **v2**：改为 18 并给出公式。

## 7. 索引构建时机：不在 prefill 末，而在 decode 初期两阶段（已修正）

- **v1 白皮书**：§6.3 标题"Index build (once, at prefill end; per layer)"。
- **代码事实**：chunked prefill 下"最后一块"从 backend 不可判定，且校准
  查询尚不存在（prefill 查询未 absorb）。实际为两阶段
  （vestigekv_mla_backend.py:1485-1732）：第一个 decode 步**同步**建
  provisional 索引（V=单位阵前 r 行、门全开、zp=Z_MAX，宁多取不漏）；
  收集 8→64 步真实 decode 查询后在**侧流异步**建 calibrated 索引并原子安装；
  已校准层捐赠基（v_init 复用——证书对任意行正交基成立）。
- **v2**：标题与开头改为两阶段描述。

## 8. "prefill 冻结、永不过期"的措辞与 close 重平衡（已修正）

- **v1 白皮书**：Eq.(0) 标注 kept "frozen at prefill"；Corollary 2 称
  "computed once ... never goes stale"；§2.2 提到 "constant-m rebalance"。
- **代码事实**：NoPE 平稳性保证的是"任何时刻计算都精确"，但实现上每个
  block close（每 4096 个 decode token）都做**全局重平衡**：
  m = round(ρ·closed) 随闭合行数增长，全局 top-m 重选，行可双向移动
  （老 kept 行可被降级入档）；kept 表重写为 [选中闭合行 | 未闭合尾]；
  召回索引只刷新 membership（_arch_idx/arch/kept_slots），**不重校**
  zp/θ_g（证书对任意 membership 成立）。
- **v2**：Corollary 措辞改为"可在任意时刻计算"；§2.2 改为
  proportional-m rebalance；新增 §7"Decode-time block close and
  rebalance"完整描述该流程。

## 9. Sketch 基的构造：eigh（协方差特征分解），不是 SVD 调用（已精确化）

- **v1 白皮书**："take the top-r right singular vectors (SVD_r)"。
- **代码事实**：对中心化校准查询的协方差矩阵做 `eigh` 取 top-r 特征向量
  （数学上等价于右奇异向量）；注释记录实测依据：eigh 17.4→4.3ms、
  子空间对齐 1.0000，svd_lowrank 仅 0.87 被拒（recall_tier.py:204-233）。
- **v2**：item 2 改为 eigh 形式并保留等价性说明。

## 10. qres 的计算：勾股恒等式，无回投影 GEMM（已补充）

- **代码事实**：因 V 行正交，q_res = sqrt(‖q‖² − ‖qsk‖²)（勾股定理），
  回投影 GEMM 被消去（fused_prologue.py:14-16, 296-305）。
- **v2**：在证书权重条目内补充该恒等式。

## 经核对一致、未改动的要点

- 几何：行宽 576 = 512 内容 + 64 sidecar；ATTN_SCALE = 192^-0.5；
  sinks=4；ρ=1/32；r=64；CLOSE_BLOCK=4096；τ=0.90；Z_MAX=8；
  门自关阈值 60%；校准窗口 8→64；FETCH_WIDTH_UNCAPPED=4096
  （实测最坏 fire 3983）。
- 层数：Kimi-Linear-48B 27 层中 7 层 full MLA（config.json 的
  linear_attn_config.full_attn_layers = [4,8,12,16,20,24,27]，1-indexed）。
- 七个图内融合 kernel：prologue 2（scores split-NK + merge/熵门/投影）
  + scan 1 + compaction 2（scan counts + write）+ CSR pack 2（prep + gather）。
- stale-by-one（scan 用上一步 query）、CSR 契约、pad/trash 行约定、
  fire 判定的逐头 max1 竞争语义、fetch 并集与 topj 截断语义、
  99.8% 触发率等引用实测数据均与代码/记录一致。

—— 冻结于 2026-09-12，对照 commit e44b914a7（branch vestigekv，
与 vestigekv-dev 树一致）。

## 追记（2026-09-12，代码侧变更）：显式激活阈值 L* 落地

v2 §"Prefill and lifecycle" item 1 原文只写"低于某激活阈值时 selection 关闭、
与 stock 逐字节一致"，未给数值。本轮代码新增了显式阈值
`ACTIVATION_MIN_TOKENS = 32768`（8×CLOSE_BLOCK）与空归档早退/grid-stride
kernel 优化（动机与实测见 `PAPER_VS_CODE_zh.md` 差异 7 的"现版"条目），
白皮书该条目同步补上阈值数值与跨阈值一次性补齐闭合的等价性说明。
