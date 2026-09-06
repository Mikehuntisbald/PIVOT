# 正例监督覆盖 × 监督目标：MM-GDINO 冻结头对照

状态：协议已封存，训练已启动；尚无新结果。v6/v7 的 checkpoint、records、统计与论文保持原样。本文档描述新干预，不把训练覆盖线索提前写成已验证机制。

## 科学问题与固定矩阵

增加复杂正例的直接监督，能否缓解 correct-output confidence 的 C–N 排序损失和完整输出风险？只使用已有 FineCops positive-source MM-GDINO-T，固定 Native 框与 Global-max 读出。

| Positive supervision | Exists E | Correct-output Y |
| --- | --- | --- |
| 全部 L1，unique-positive uniform | seeds 17/42/73 | seeds 17/42/73 |
| 全部 L1/L2/L3，unique-positive uniform | seeds 17/42/73 | seeds 17/42/73 |

共 12 个新头。旧 paired-L1 E/Y 是封存参考，非这次 difficulty-coverage 主对照。旧 loader 使用 43,979 个 L1 parents，按 negative-edit 数重复加权，还遗漏 10,036 个 L1 正例；新增 L1-uniform 对照避免将这些变化都归给 L2/L3。

本实验改变的是正例监督人群；Full 的图像/表达构成及 L1 相对权重也随覆盖改变。它不是难度标签的独立因果干预，不证明某个语言复杂性因素是唯一原因。官方 positive difficulty 不等于 negative edit level 或纯句法复杂度。

## 采样与训练预算

- 正例池：L1=54,015；L2=25,282；L3=4,044；Full=83,341。先按 sample ID 排序。
- 独立 `PCG64(SeedSequence([20260912, training_seed]))` 生成连续无放回 permutation cycles；不在每个 negative epoch 重置正例流。两个 target 使用同一个流。
- 每头 402,255 次正例呈现：L1 每个正例 7 或 8 次，Full 每个正例 4 或 5 次。全部 unique positives 均进入 loss。
- 负例复用原 `_epoch_schedule`：保留生成并跳过 rank events 的随机数消费，80,451 negatives 每轮恰好一次，共五轮 402,255 次。每轮 2,514 个 B32 和末批 B3，总计 12,575 次更新。
- B32 在这里表示 32 positive + 32 negative；新 batch 不再声称是 32 个语义配对。BCE/L2 分别对两个 source 求均值，没有 paired margin、BatchNorm 或跨记录注意力。
- 同一原始初始化、50,179 个可训练参数/8 tensors、FP32 deterministic、AdamW LR=1e-4、WD=0、clip=0.1、logit L2=1e-3。正负 source 权重各 0.5，不按 E/Y 重平衡标签。
- trunk 不加载到 optimizer，Native rank owner 永久冻结。每个 seed 的四头参数和 optimizer state 独立；无新增 detector traversal。
- 五轮是原 negative 数据的五个 epoch，不是“全体 83,341 positives 完整遍历五遍”。

冻结特征使用校验 SHA 后的只读用途 mmap loader，保持原 FP16 features、FP32 scores/boxes 和900 queries；三个 worker 共享文件页，避免三份约75GB特征堆叠突破远端200GiB主机内存限制。旧 cache loader 与新 loader 的 tensor parity 已由测试验证。

## 评价与结果规则

全部 12 个 U12,575 终点及成功进程退出封存后才评分。复用既有 FineCops val/gRef cache，所有 Native boxes、scores、correctness、candidate-mask SHA 必须与 v6 records 一致。旧 paired-L1 score 直接复用。

评价面固定：

| Surface | Positive | No target | Images |
| --- | ---: | ---: | ---: |
| FineCops val | 9,426 | 9,029 text negatives | 3,567 |
| gRef Full TestAB single/no-target | 11,563 | 9,121 | 1,500 |
| gRef FineCops train+val source-disjoint | 9,848 | 7,716 | 1,277 |

不打开 FineCops Test；不增加 negative-image 或 multi-target；不选 checkpoint、阈值或配方。

主指标 `mixed AUGRC`：

`D_Y = R(Full,Y) - R(L1,Y)`；`D_E = R(Full,E) - R(L1,E)`；`I = D_Y - D_E`。

同时报告四格原始风险、Full 下残余 Y−E 风险差、每 seed、mean/sample SD、C–W/C–N/W–N、positive-difficulty C–N、AURC、existence AUROC 和诊断 FPR95。旧 paired-L1 对照另列，不能代替新矩阵。

5,000 次 paired image-cluster bootstrap，PCG64 seed20260912；同一 draw 同时应用四格、旧参考和三个 seed，gRef 按 TestA/B 分层。每 seed 重算指标再等权平均，FPR95 每轮重算 replicate-positive q05，不拟合部署阈值。区间是固定三头平均的 image uncertainty，不是 n=3 训练 seed 总体保证。

只有 interaction 变负，不能称为修复 Y；还要看 Y 的绝对风险和 C–N 变化。区间跨零不能直接称为消除了损失。结果解释以效应幅度、精度、残余损失和迁移为准，不要求任何 arm 获胜。

## 执行与产物

远端根：`/mnt/why/PIVOT/outputs/arrow_confidence_coverage_v1_20260906`。

协议 SHA-256：`f208df694a6ca6eef256de38f463b641e08f422fcd77c1dc380f4859c0150843`。

GPU0/1/2 分别训练 seed17/42/73 的四头；全部完成后 GPU3 执行缓存评分，再并行三个 CPU bootstrap。所有失败保留，不能把进程启动或 checkpoint 存在当成完成。

入口：

```bash
.venv-b32a1/bin/python tools/lock_confidence_coverage.py \
  --parent outputs/arrow_confidence_readout_v6_20260905/protocol.json \
  --output outputs/arrow_confidence_coverage_v1_20260906

.venv-b32a1/bin/python tools/run_confidence_coverage_stage.py \
  --protocol outputs/arrow_confidence_coverage_v1_20260906/protocol.json \
  --localizer mmgdino_positive \
  --train-cache outputs/b32a1_finecops_positive_trunk_factorized_20260904/formal_cache/train/manifest.json \
  --val-cache outputs/b32a1_finecops_positive_trunk_factorized_20260904/formal_cache/val/manifest.json \
  --gpus 0 1 2 --detach
```

以上是已使用的命令，不应重复启动。`heads/*/seed*/postflight.json` 是逐 seed 验证，`all_heads_sealed.json` 是评分前门槛，`completion.json` 才是三套统计的终态入口。

预训练验证：远端 21 tests passed，包括 GPU3 synthetic B32/900-query U1、参数隔离、Native 不变、单步与 resume 基础逻辑、两路 cache parity、完整流覆盖、原 negative 顺序和末批、bootstrap determinism、交互解释反例。没有在这些 synthetic 测试上选择参数。

部署记录：远端无 rsync，首次传输未成功且未训练；随后只用 scp 复制本次新增文件。所有 v6/v7 实验源码与旧产物未修改。

## Figure 1(b) 的 seed 显式修订

独立的 [v7.1 主文](../paper/empirical_study_v7_1.tex) 保留 v7 原版本，数值仍来自 sealed v6，不混入尚未完成的 coverage 结果。

| MDETR FineCops AUGRC change ×100 | Seed17 | Seed42 | Seed73 |
| --- | ---: | ---: | ---: |
| G-trained, inference G→S | +4.725806 | −0.060922 | +0.157665 |
| Matched S training/deployment − G/G | −0.349355 | −0.074748 | −0.143133 |

Figure 1(b) 按 seed 连线，黑色 diamond 保留均值；不再只展示大均值与窄 image CI。图注、Abstract/Introduction、对应正文明确 seed17 驱动 +1.608 的均值，推理切换的 effect sample SD=2.703。这是 seed-dependent sensitivity，而非跨 seed 稳定的空间修复机制证据。
