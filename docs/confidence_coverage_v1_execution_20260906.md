# Coverage-v1：12 头训练与评分已完成，bootstrap 运行中

这是 [固定协议](confidence_coverage_v1_20260906.md)之后的执行快照，不是新结果报告。

- seeds17/42/73，每 seed 四头，全部完成 12,575 updates / 5 个 negative epochs。
- 三个训练进程 returncode=0；全部 12 个 endpoint 在评分前封存。
- 全部轮次 zero optimizer/AMP skips、zero nonfinite；rank owner 与初始 frozen SHA 一致，每头只有 confidence 的 8 tensors / 50,179 parameters 更新。
- FineCops val 和 gRef Full 缓存评分全部完成，所有新头共用的 Native boxes、scores、correctness 和 candidate-mask 与旧 v6 逐记录一致。
- 全部 GPU 已释放，三套 5,000 次 CPU bootstrap 在后台独立运行。此时未完成区间，不能宣布 coverage 假设成立或不成立。
- 没有 FineCops Test 访问、detector forward、checkpoint selection 或 threshold fitting。

## 封存终点

每个 checkpoint 同时包含 `l1_uniform__exists / l1_uniform__emit / all_uniform__exists / all_uniform__emit` 四个独立 owner。

共同目录：`/mnt/why/PIVOT/outputs/arrow_confidence_coverage_v1_20260906/heads/mmgdino_positive/`。

| Seed | Checkpoint | SHA-256 |
| --- | --- | --- |
| 17 | `seed17/checkpoint_epoch5.pt` | `633f9a75d7830df84cd4152261734b1d03d8eb646531a12b9f3665bcd80cb05a` |
| 42 | `seed42/checkpoint_epoch5.pt` | `d0d39c094a932301977f0424acb400297895421364e675fea4b58b4082f395ab` |
| 73 | `seed73/checkpoint_epoch5.pt` | `1b3168985825141102e41d84c621c6b9137d3dc7f90b451f90178a186fdeeb17` |

三 seed 记录的 peak allocated GPU memory 均约0.399GiB、最低 free约94.087GiB。这是缓存上的轻量头训练，不是 detector 训练显存，也不需要为占显存而改变公平 B32 配方。

## 可核验收据

- [协议](../paper/data/coverage_v1/protocol.json)
- [三个进程终态](../paper/data/coverage_v1/training_terminal.json)
- [评分前全部头封存](../paper/data/coverage_v1/all_heads_sealed.json)
- [seed17 postflight](../paper/data/coverage_v1/seed17/postflight.json)、[seed42](../paper/data/coverage_v1/seed42/postflight.json)、[seed73](../paper/data/coverage_v1/seed73/postflight.json)
- [FineCops val 评分](../paper/data/coverage_v1/finecops_val/postflight.json)、[gRef Full 评分](../paper/data/coverage_v1/gref_full/postflight.json)
- [进度快照](../paper/data/coverage_v1/execution_snapshot.json)

后台入口在三套分析成功后写入远端 `completion.json`，并绑定三份完整分析的 SHA；训练完成或评分完成都不代替这个最终门槛。无需再次启动训练。若分析失败，保留现有输出和日志，先解释失败，不能换配方、换数据或挑选较好的 seed。

## Figure 1(b)

[v7.1 主文 PDF](../paper/empirical_study_v7_1.pdf) 已编译为7页；旧19页补充保持不变。图中显示三 seed 的配对变化和均值，明确 +1.608 的 inference-only 均值主要由 seed17 驱动。修改后的图注、摘要、Introduction 和对应正文保持一致。

[PDF/构建收据](../paper/data/coverage_v1/v7_1_build_receipt.json)与[PDF 检查](../paper/data/coverage_v1/pdf_validation_v7_1.json)单独封存；它们没有包含待完成的 coverage 结果。
