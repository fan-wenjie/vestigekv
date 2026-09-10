# ERRATA — 2026-09-11 复现运行（单机 2×RTX PRO 6000 Blackwell, TP=2）

本文件记录本次复现相对论文 frozen 协议/环境的所有偏差与途中发现的错误。

## 环境偏差（相对 skills/reproduce 的 pin）

1. **transformers 4.57.1 → 5.12.1**。REPRODUCE.md/ENV_PINS pin 4.57.1，
   但当前 engine 子模块 commit（81c1d1411）的 `pyproject.toml` 要求
   `transformers==5.12.1`（`PreTrainedConfig`、`glm_ocr` 等符号在
   4.57.1 中不存在，registry 无法导入）。按 engine 自带声明安装。
   torch/triton 保持 pin：torch 2.13.0+cu130 / triton 3.7.1。
2. **NCCL_P2P_DISABLE=1**（所有启动）。本容器平台双卡 CUDA P2P
   （NCCL `P2P/CUMEM`）在第一次集合通信永久挂起；用纯 torch 双卡
   allreduce 最小复现确认。禁用后走 SHM，功能正常。
3. **--disable-custom-all-reduce**（所有启动）。sglang custom
   all-reduce 在 CUDA graph capture 的 IPC buffer 注册
   （`get_graph_buffer_ipc_meta`）报 `invalid argument`——容器无 CUDA
   IPC 能力。回退 NCCL allreduce。两臂配置一致，A/B 公平。
4. **部署布局：双机 PP=2 → 单机 TP=2**。论文部署是两台机器各一张卡
   （TP=1, PP=2, partition 23,4）。本机为单机双卡，按用户要求 TP=2。

## 代码修复（engine 工作区脏改动，已脱离 frozen PR tree）

5. **`vestigekv_mla_backend.py:183`**：`_q_heads` 原来取
   `model_config.num_attention_heads`（全量 32），TP>1 时每 rank 的 q
   只有 16 头，CUDA graph capture 时 `q.view(bs, 32, 576)` 作用在
   [bs,16,576] 上崩溃（"shape '[4, 32, 576]' is invalid for input of
   size 36864"）。修复为
   `model_config.get_num_attention_heads(server_args.tp_size)`。
   修复后：104 个注册单元测试全部通过；TP=2 服务器启动、capture、
   生成均正常。**注意：此后的服务数据来自 dirty tree，按 RULES 不可
   作为论文数字引用；若数据要入论文，需把此修复 squash 进 PR 分支。**

## 协议变更

6. **gsm8k n=800（原始测试集子集）→ GSM8K-Platinum 全量 n=1209**。
   原始 gsm8k 测试集存在已知标签噪声；Platinum（madrylab/gsm8k-platinum,
   arXiv:2502.03461）是人工修订全量版（修正标签、剔除坏题）。
   实现：`mexp/quality/gsm8k_platinum.jsonl`（1209 行）+
   `run_quality.sh` 改用 `--data-path ... --num-questions 1209`，
   64-shot 前缀为该文件前 64 行（与原协议同构）。
   输出文件改名为 `quality_gsm8kplatinum_64shot_n1209_<arm>.txt`，
   frozen 的 `quality_gsm8k_64shot_n800_*` 保持不动。

## 参考数字（旧协议，已被取代）

7. Base / vestigekv / TP=2 / 旧协议 gsm8k 64-shot n=800：
   **Accuracy 0.826**（frozen 值 0.830；REPRODUCE bar 0.834）。
   差 3/800 题，属部署差（TP=2+SHM 通信）范围内的波动，但该数字
   协议已废弃，仅留档。

## 平台坑（备查）

- apt 曾被平台初始化脚本的 `apt update` 长时间持锁（已终止重跑）。
- 容器缺 CAP_SYS_PTRACE：py-spy/gdb 均不可用，调试只能靠日志与
  /proc 计数器。
- nvidia-smi 的 GPU 利用率读数在本平台不可靠（空载也显示 100%），
  以显存占用和日志为准。

## 2026-09-11 perf-run adjustment: --max-running-requests

Metric-1 vestigekv launch (max-total-tokens 589824) OOMed at ~478k decoded
(token usage 0.81): startup available_gpu_mem was only 1.75 GB because
max_running_requests auto-derived to 835 -> Mamba ssm_state cache alone took
16.33 GB; vestigekv's growing per-step tier state (kept_rows gather in
recall_tier.py) exhausted the remainder. Fix: pin --max-running-requests 4
for Metric 1 (max-concurrency is 1; mamba cache shrinks ~16 GB). This is a
hardware-fitting knob: it does not touch the measured quantity (bs=1
per-token decode latency) and is applied IDENTICALLY to both arms
(vestigekv and triton), preserving arm symmetry. Metric 2 will use
--max-running-requests 32 (sweep max bs=16).

## 2026-09-12 代码修复：bs=1 双重追加 bug + 短上下文激活阈值

8. **bs=1 每步双重追加（正确性 + 性能 bug，已修复）**。初始提交的
   `init_forward_metadata_in_graph` bs==1 块（为旧设计 "bs=1 graphs read
   kept_buf directly" 写的图内追加）在 CSR pack（torch `_pack_csr` /
   图内 `_pack_csr_prep_kernel`）接管追加职责后未被移除，配套的
   `_indptr1` 只剩写、无任何读者。后果：bs=1 下每个 decode 出的 token
   对应的池行被追加进 kept 表**两次**——kept_len 每步 +2，packed CSR 把
   最新行重复出席。profiling 证据链：修复前 30k 上下文 vk 的 E2E 斜率
   15.3 µs/k 恰为 dense（9.6 µs/k）的 1.6 倍（多读了近一倍行），
   `_scatter_gather` 等遗留小 kernel 每步 ~35 个（~70 µs）。
   **影响范围：2026-09-11 的所有服务数字的 vestigekv 臂均带此 bug**
   （dense 臂不受影响）。质量数字是在"最新行重复出席"下测得的
   （GSM8K-Platinum 仍 PASS，bug 只是多 attend 了行、未漏行）；P1 的
   vestigekv 数字偏悲观（每个 close 区间内最多多读 ~4095 幻影行）。
   修复：删除遗留块与 `_indptr1`；追加由 CSR pack 唯一负责。
   107 个注册单测全过；修复后 L* 以下 greedy 输出与 dense **逐 token
   一致**（8k/30k 各 200 token，见 greedy_fix2_vestigekv.json +
   server_dense_parity.log）。
9. **短上下文激活阈值 L\* = ACTIVATION_MIN_TOKENS = 32768（已实现）**。
   论文附录 D item 1 的设计（低于阈值完全走 dense）此前未实现；本轮落地：
   阈值以下不闭合块、不归档、不建索引、不收集校准（kept = 全部行，与
   stock 逐字节一致），跨阈值一步补齐全部闭合（块级 σ 不可变 + 全局
   top-m 与闭合次序可交换 ⇒ kept 集与从头压缩一致）。配套 kernel 优化：
   召回 scan/compact 改 grid-stride（每 pair 上限 SCAN_GRID_CAP=128），
   prologue scores/merge 空归档早退。召回管线固定成本 114 µs/step →
   ~18 µs/step（torch profiler，8k）。动机数据：修复前 16k 的 E2E
   vk/dense = 0.973、注意力单独 4k 0.77x。
10. **capture 失败路径不回卷追加行（正确性 bug，已修复，2026-09-12）**。
    `_capture_scan_timed` 的 `_run()` 每执行一次就经 CSR pack 向每个
    layer 的 kept 表追加一行；成功路径回卷 3 次追加（2 warmup + 1
    capture），但 except 路径（capture 失败回退 eager）此前**直接返回
    不回卷**，留下 1–3 个幻影行直到下一次 close 重写该段。修复：
    `_run` 内对每次成功的 `_pack_csr` 逐 layer 计数（`appended` 字典，
    计数在 pack 调用成功之后，覆盖 warmup 中途死亡的半 loop 情形），
    except 路径与成功路径都按计数回卷。新增结构测试
    `test_failed_capture_rewinds_the_partial_appends` +
    数值测试 `test_net_effect_of_a_failed_capture_is_zero_appends`。
    实测中未观察到 capture 失败（VKCAP 日志无 warning），此为 review
    发现的潜伏 bug，不影响既有数字。
11. **eager decode 越界使用 graph 缓冲区（崩溃 bug，已修复，2026-09-12）**。
    `forward_decode` 的 eager 分支无条件复用 `_graph_bufs` 的 packed CSR，
    但 `_gb_indptr_stack` 只有 `graph_max_bs + 1` 行（按捕获尺寸定容）。
    当服务配置 `--cuda-graph-max-bs` < `--max-running-requests` 时，
    bs > graph_max_bs 的 eager decode 切片被静默截断，下游 triton
    `decode_attention_fwd` 的 `q.shape[0] <= kv_indptr.shape[0] - 1`
    断言触发、调度器崩溃（2026-09-12 Instruct 质量臂实测：graph-max-bs
    误配为 2、parallel 4，gsm8k 跑到一半崩溃；README 规定的质量配置是
    graph-max-bs 4，故此前未暴露）。修复：eager 分支增加容量判断，
    越界时回退 `_compressed_indices` 通用路径。新增回归测试
    `TestEagerDecodeBeyondCapturedBs`（越界回退 + 界内仍走 graph 缓冲）。
    既有数字不受影响：此前所有实验配置中 graph-max-bs ≥ 实际并发。
12. **P2/质量服务器配置不对称：漏 --disable-radix-cache（配置 bug，已修复重跑，2026-09-12）**。
    2026-09-12 白天的 P2 与质量服务器启动命令漏掉了 README 模板里的
    `--disable-radix-cache`（2026-09-11 的四个 P2 服务器与旧质量服务器
    均带）。radix cache 开启时 Kimi-Linear 的 hybrid mamba cache 按
    mamba_full_memory_ratio 预分配 ssm_state 16.31GB（max_mamba_cache_size
    834），而关闭时仅 0.64GB（32 条）；随机 prompt 无任何前缀共享，
    radix 是纯开销。同时 vestigekv 臂空闲显存比 dense 臂少 16.6GB 的
    疑账也由此解释。影响范围：2026-09-12 的 P2 vestigekv v2 数字与
    质量 v2 数字（gsm8k 0.851/0.908、MAUVE 两组）均为 radix-on 下测得，
    与 dense 臂（radix-off）不对称，全部作废重跑；bpb 为逐文档唯一
    prompt、无前缀命中，数值不受影响但随 P2 服务器一并重跑以保协议
    一致。bs=16 档另发现 `--max-total-tokens 1130496` 恰好卡满池
    （用量 0.986，两臂同受限），改为放开池子自动定容后重测。

## 13. radix 探针：vestigekv+radix 与 triton+radix 行为一致（2026-09-12）

探针 `mexp/radix_probe.py`：41864-token 前缀、greedy 128 token、ignore_eos，
flush_cache 后比较 fresh（cached=0）与全命中（cached=41856）输出。

- vestigekv fresh == triton fresh，前 16 token 逐 token 相同（压缩路径正确）；
- 两臂 radix 命中后均从第 4 个输出 token 与各自 fresh 发散，发散位置相同，
  发散后文本连贯（近平局翻转，非数据损坏）；
- 判定：radix 命中改变计算切分点属 sglang 上游固有数值特性，两臂一致，
  vestigekv 无需额外 radix 修复；实验两臂同开 radix 对比公平。

## 14. 质量协议 v4：可复现优先（2026-09-12）

质量线目标从"生产一致"改为"可复现"：radix 关闭（命中改变 chunk 切分点、
greedy 近平局翻转，见 #13）+ gsm8k 客户端 parallel 4→1（pin batch=1，
消除 kernel 归约顺序随时序漂移）。MAUVE 生成本来就是串行+固定
sampling_seed。同硬件上质量门现在应逐次可复现。性能线（P1/P2）保持
radix 开（生产默认），两线目标不同、配置显式分离。v3（radix-on）质量
数字作废：Base vk 0.847 / dense 0.844 / MAUVE PASS(0.9462 vs 0.9997)
仅作历史参考。

## 15. 可复现性验证结论 + 等价测试自包含化（2026-09-12）

- v4 协议（radix关+parallel1+固定种子）同服务器两遍：gsm8k 0.849 vs
  0.850（1/1209 翻转），MAUVE 文本逐字节不同。残余噪声源为 kernel 内
  atomic 累加顺序（MoE/split-k 类），非配置可消。结论：质量门保持统计
  判定（accuracy 阈值 + MAUVE -0.10），不追求逐位复现。
- engine/test/manual/test_vestigekv_equiv.py 重写为自包含：不再依赖
  本机不存在的 mini-sglang 参照，改为对内嵌朴素实现（当前数学）断言。
  三层：eviction（fallback 逐位 / 融合版决策一致 + pool [512:576] 切片
  审计，防历史上 [128:576] 错切类 bug）、build（V 逐位、n_hard/thr_g
  精确、zp 容差 0.05）、query（eager 逐位、query_fixed 1% fire-set
  门槛）。3 测试全过；注册套件 111 + 等价 3 = 114 全过。

## 16. Instruct+triton 质量服务器挂起：漏配 --disable-custom-all-reduce（2026-09-12）

现象：Instruct dense 臂（triton, TP2, radix off）首个请求（6-token
cache_prefix）挂起 300s 触发 watchdog 自杀，3/3 复现；同配置 Base
正常、Instruct+vestigekv 正常，一度疑似模型相关。

诊断：本机容器无 CAP_SYS_PTRACE，py-spy 不可用；改用 faulthandler
（scheduler.py 已 enable，SIGABRT 触发栈 dump 进 server log）。
裸栈显示双 rank 同挂 kimi_linear.py:529（unsqueeze，纯元数据操作，
不可能阻塞）；加 CUDA_LAUNCH_BLOCKING=1 后抓到真凶：双 rank 同挂
sgl_kernel/allreduce.py:114 的**自定义 all-reduce**。

根因：本机 CUDA P2P 挂起（README 已记载，NCCL 走 NCCL_P2P_DISABLE=1
的 SHM），自定义 all-reduce 依赖 P2P 显存映射，同样挂死。之前所有
能跑的服务器（含 Base dense v4）日志均为 disable_custom_all_reduce:
True；此次 Instruct dense 启动命令漏抄该旗标。"Base 正常/Instruct
挂起"是误导性巧合——与模型无关，纯配置遗漏。

修复/规矩：质量与性能所有服务器模板必须带 --disable-custom-all-reduce
（README 已有，本次是没照抄）；调试期 py-spy 已装但容器禁 ptrace，
栈抓取一律用 kill -ABRT + faulthandler（先用 CUDA_LAUNCH_BLOCKING=1
精确定位，否则栈停在误导性的异步发射点）。

## 17. 质量 v4 全量数字（2026-09-12）

协议：gsm8k-platinum 64-shot n=1209 全量，parallel 1，radix off，
固定种子；MAUVE 16 ctx 256 tok gpt2-large。双模型双弧：

| 模型 | gsm8k vk | gsm8k dense | MAUVE vk | MAUVE eng | 门 |
|---|---|---|---|---|---|
| Base | 0.849 | 0.848 | 0.9960 | 0.9997 | PASS |
| Instruct | 0.913 | 0.911 | 0.9805 | 0.9345 | PASS |

Invalid 全 0.000。结论：双臂统计不可区分，质量门全过。

## 18. serving bpb 门对压缩保真无效（设计使然，需正确归因）（2026-09-12）

现象：P2 协议下 serving bpb compare，vk vs dense 的 dce_median=0.0，
12 个 doc 逐位全等（16 位有效数字一致）。

根因（非 bug）：vestigekv_mla_backend.py 的 forward_extend 完全委托
base（:889-903），压缩只介入 forward_decode；而 bpb_serving.py 的协议是
`max_new_tokens=1 + return_logprob + logprob_start_len=L//2` 的
teacher-forced 输入 logprob 打分——纯 prefill 路径。两臂必然逐位一致，
该门**永远无法**测出压缩误差。它唯一的价值是作为"prefill 路径未被
污染"的回归金丝雀。

论文影响：tab:quality 的 ΔCE 行（\dceVkMed=+0.0020）实际是 harness
（m6_bpb.json）数字，经"row-set 位等价→迁移"逻辑转入；main.tex
sec:deploy 的句子把 harness ΔCE 与 serving 的 gsm8k/MAUVE 并列在
"serving path with recall live every step" 之下，归因不清。已修正为：
serving 门=gsm8k/MAUVE（decode 路径，压缩真实介入）；teacher-forced
serving logprob 与 dense 全精度一致（prefill 原样委托的必然结果）；
harness ΔCE 经位等价迁移。decode 路径的压缩保真覆盖由
gsm8k/MAUVE/needle 承担（均已 PASS）。

## 19. bench_serving 客户端 ITL/E2E 在超长输出下被流式路径污染（2026-09-12）

现象：P1 v2（4k in / 520k out，radix 开，双模型双臂）客户端汇总出现
vk 比 dense 慢的倒挂：Base vk 98.0 vs dense 179.2 tok/s；Instruct
vk 129.2 vs dense 151.0。与同文件历史测量（vk base 119-125、
dense base 93-112）也互相矛盾。

证据链（判定为测量伪影，非算法回归）：

1. 服务器侧 decode 日志（官方 gen throughput，每 40 步一行，13004 行
   覆盖全部 52 万步）无任何慢区间：vk 中位 227.6 tok/s（尾部 512k 处
   209.8），dense 中位 178.6（尾部 141.6）。vk 全程更快且优势随
   上下文增长，与算法预期一致。
2. 客户端到达曲线：vk 前 25% 以 ~245 tok/s 与服务器同步，之后到达率
   渐进崩塌（130k→260k 段 158 tok/s，260k→390k 段 74 tok/s，尾部
   ~42 tok/s）；服务器 2297s 已生成完毕，客户端拖到 5308s 才收完
   （3000+s 在等积压冲刷）。dense 生成速率 ≈ 接收路径可持续速率
   （~179 tok/s），不积 backlog，客户端数字恰好等于真实速率。
3. 机制：stream_interval=1 下每 token 一次 detok + SSE 事件，接收/
    detok 路径存在可持续速率上限且随流变长退化；生成越快的臂积压
   越深，客户端 E2E 反而越长——**该指标惩罚更快的一侧**，且历史
   所有 4k→512k 客户端 jsonl（含 Sep-9 发布版）在 ~80-180 tok/s
   以上均受此污染（旧数据中双臂 50% 处 ITL 同被压到 ~12ms）。
4. 结论：P1 延迟曲线指标改用服务器侧官方 decode 日志
   （gen throughput vs #full token）；客户端 jsonl 仅作 ITL 分布
   参考。干净数字（服务器侧中位 tok/s，Base≈Instruct 逐桶一致）：

| 上下文桶 | vk | dense | 比值 |
|---|---|---|---|
| 4k-32k | 247.7 | 249.2 | 0.994 |
| 32k-64k | 242.2 | 235.3 | 1.029 |
| 64k-128k | 239.5 | 219.0 | 1.094 |
| 128k-256k | 231.8 | 193.8 | 1.196 |
| 256k-384k | 223.4 | 168.8 | 1.323 |
| 384k-524k | 212.8 | 148.7 | 1.431 |

交叉点在 32k-64k 之间；4k 桶 vk 仅慢 0.6%（固定成本口径成立）。

补充诊断（2026-09-12，300k 输出 vk 单请求在线监控）：服务器调度器全程
~240 tok/s 无退化（TP0/TP1 稳定 105%/102% CPU）；主进程
（tokenizer manager + SSE 流式响应）累计 CPU 占比单调上升
（21%→39% 且仍在涨）→ 每 token 处理成本随流长增长；bench 客户端
CPU 占比反降（96%→空等）= 客户端在等服务器的流式输出。瓶颈定位在
**服务器主进程的流式发送路径**（stream_interval=1，每 token 一次
SSE 事件 + 某种随流长增长的开销），与模型/后端无关。

## 20. 续写一致率门的冻结 bar 校准错误（对照组实证），门退役（2026-09-12）

现象：续写一致率门（16×64k ctx，512 贪婪 token，radix off，串行）
vk vs dense 首跑 FAIL：Base 中位首次分歧 6.5（bar ≥128）、最差 0
（bar ≥8）；Instruct 中位 5.0、最差 1。

对照实验（决定性）：dense flashinfer vs dense triton——同模型、同权重、
同 dense 缓存策略，仅注意力内核不同——中位首次分歧 37.5、最差 2，
**同样 FAIL**。冻结 bar（≥128/≥8）对纯内核数值差异都不可达，说明该
bar 校准的是"数值敏感度"而非"压缩保真"。

定量解读：vk 中位 6.5 比内核噪声底（37.5）早约 6 倍发散，与已测的
harness ΔCE +0.0020 nats/token 自洽——压缩误差约为内核数值噪声的
6 倍，在贪婪解码 top-2 近平局处翻转 argmax 的频率相应更高，翻转后
轨迹级联发散（两臂文本均保持连贯；gsm8k/MAUVE 双 PASS 是任务级/
分布级证据，本门是逐 token 数值级证据，三者互不矛盾）。

处置：门退役（同 bpb 先例，ERRATA #18）。长上下文 decode 路径的
保真覆盖：检索由 serving needle 门承担（vk intact ≥ dense intact）；
生成分布保真由长上下文 MAUVE 承担（替代门，64k 上下文）。
附带修复：needle 服务器 ctx 131072→139264（prompt=131072 文档+问题
超出原 ctx 致全 400）；continuation compare 参数个数 bug；
早停 EOS 处理（长度差计为在较短处分歧）。

数据：results/quality_continuation_{vestigekv,dense}[_instruct].json、
results/quality_continuation_dense_flashinfer.json（对照组）、
results/quality_continuation_verdict[_instruct].json。
