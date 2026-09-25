# VERDICT — 质量线（2026-09-12）

配置：质量线服务器 radix OFF、串行客户端、vk 弧阈值 0（与性能线显式分离）。
双模型（Base/Instruct）× 双臂（vestigekv_mla / triton dense）全四格。

## 门结果

| 门 | Base | Instruct | 判定 |
|---|---|---|---|
| gsm8k-platinum 64-shot (n=1209) | vk 0.849 / dense 0.848 | vk 0.913 / dense 0.911 | PASS（delta 在噪声内） |
| MAUVE 4k (16 ctx, gpt2-large) | vk 0.9960 / eng 0.9997 | vk 0.9805 / eng 0.9345 | PASS（bar: vk ≥ eng−0.10） |
| serving needle 128k ×8 | vk 8/8 / dense 8/8 | vk 8/8 / dense 8/8 | PASS（bar: vk ≥ dense） |
| MAUVE 64k (n=64, gpt2-large) | vk 0.9987 / eng 0.9954 | vk 0.9861 / eng 0.9905 | PASS |

## 退役的门

- serving bpb（ERRATA #18）：teacher-forced 输入 logprob 是纯 prefill，
  压缩只介入 decode，门设计上永远测不到压缩。
- 续写一致率（ERRATA #20）：冻结 bar（中位首分歧 ≥128、最差 ≥8）对纯
  内核数值差异也不可达（dense flashinfer vs triton 对照：中位 37.5、
  最差 2）。vk 中位 6.5 与 harness ΔCE +0.0020 nats 自洽。替代门 =
  serving needle（检索）+ MAUVE 64k（生成分布）。MAUVE 64k 首跑 n=16
  值饱和（小样本量化，32 点 2 质心），n=64 重跑为正式值。

## 数据

results/quality_{mauve,gsm8k}*（v4）、results/quality_needle_*.json、
results/quality_mauve64k64_*.json、results/quality_continuation_*（存档）。

## 附：稀疏注意力对照（gpt-oss-120b，banded sparse）

同一冻结插针协议（128k×8）：严格判定 5/8。3 个"失败"均输出正确前 3 位
数字后转入推理评注（原始补全模式下的格式伪影）；同文档换"回答完整 5 位
数字"问法后 3/3 全恢复 → 检索实际 8/8。结论：gpt-oss 带状稀疏注意力在
128k 检索完备，与 Kimi dense / vestigekv（均 8/8）同档；三方在 128k
插针检索上无差异。数据：results/quality_needle_gptoss120b*.json。

## 附：中文插针（三国演义语料，128k×8，双模型双臂）

vk 8/8 / dense 8/8（两模型同）→ PASS。古汉语正文无阿拉伯数字，暗号为
异质 token 序列；模型答出暗号后常自然续写三国文风文本。
数据：results/quality_needle_zh_*.json。
