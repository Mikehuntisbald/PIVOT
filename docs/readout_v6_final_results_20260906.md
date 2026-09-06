# v6 完整结果：监督目标改变失败排序，但读出改进与完整风险收益具有条件性

状态：**18 个新头、两个冻结 localizer、三个评估面与全部 5,000 次统计已完成**。本文件只解释已封存数值，不增加训练或选模。实验完成不等于论文构建／投稿准备完成；后者由独立 build receipt 记录。v5、阶段性结果和旧实验谱系保留。

## 1. 最重要的结论

**同一个监督目标效应，可以对应相反的完整风险结果。** 在 FineCops val，Global emit 相对 exists 都提高定位正确性排序、降低 existence AUROC；但 MM-GDINO 的 mixed AUGRC 增加 **+0.148624 [0.058134, 0.235490]**，MDETR 则降低 **−0.865560 [−0.987536, −0.744694]**。因此，不能把 existence AUROC 的下降直接解释成成功输出排序变差，也不能把第一模型的结论推广成“直接监督 emit 总是有害”。

Selected-query 没有成为通用修复：MM-GDINO 上没有确认主要指标的 emit 改善；MDETR 在 FineCops 的两种目标几乎等幅改善，但冻结迁移到 gRef 后，Selected 的 emit 风险明确上升。**有证据的是监督目标、读出设计和评价人群的交互及其边界，不是已证明的唯一空间原因。**

固定组合也不是简单的正／负结论：Product 在 MM-GDINO 的两个 gRef 面都优于 Native、G-exists、G-emit，但在 FineCops 劣于两个已学头。它给出了无需新训练的、具体设置下可复用的实践收益，不证明普遍可组合或不可组合。

## 2. 范围、训练覆盖与统计口径

两个 localizer 分别是 FineCops positive-source MM-GDINO-T 和官方 RefCOCO MDETR-R101 EMA。比较的是**各 localizer 内部**的目标／读出效应，不将两者绝对分数差归因于纯架构差异。所有框及 Native 选中 query 固定，confidence 从不重新选框；Selected 从原始初始化训练，不是从 Global 终点继续训练。

| 评估面 | 图像 | Positive | No-target | MM Native P@1 | MDETR Native P@1 |
|---|---:|---:|---:|---:|---:|
| FineCops val | 3,567 | 9,426 | 9,029 text negatives | 77.360492% | 65.563335% |
| gRef Full TestAB | 1,500 | 11,563 | 9,121 | 55.945689% | 81.838623% |
| gRef FineCops-source-disjoint | 1,277 | 9,848 | 7,716 | 56.041836% | 81.976036% |

gRef 只含 single-target／no-target，**multi-target 明确排除**，不是完整 GREC 评价。Source-disjoint 排除 FineCops train/val 的源图重叠；不宣称与所有 localizer 上游训练图像均无重叠。Full 与 disjoint 是重叠的主面／敏感性面，不是两次独立复现。没有打开 FineCops Test，没有扩展 negative-image，没有根据这些结果换 checkpoint、调读出、权重、Gap 或部署阈值。

直接 head 监督的范围必须保留：80,451 个训练 pairs 来自 **43,979 个唯一 L1 positive parents**；全部 83,341 个可用训练 positive 则为 L1/L2/L3 = **54,015 / 25,282 / 4,044**。L2/L3 正例未进入 confidence head loss。完整正例缓存、标签审计与 SIRC 统计均不等于 loss 监督；MM trunk 的早期 positive-source 训练更广。两目标／两读出的训练 population 相同，因此受控比较仍成立，但这不是同分布、无限容量下 Bayes 最优性失效的证明。详见[训练覆盖核查及源哈希](readout_v6_mm_stage_results_20260906.md#8-后续-metadata-核查confidence-的直接监督只覆盖-l1-parent-正例)。

本块属于**事后动机、事前锁定配方的机制研究**，不是 virgin held-out confirmation。每面采用 5,000 次 **paired image-cluster** bootstrap、PCG64 seed `20260911`；gRef 按 TestA/TestB 分层。同一 image draw 携带全部 expressions，并同时作用所有 localizer、读出、组合和 seeds 17/42/73；逐 seed 重算指标后等权平均。FPR95 的每轮 positive q05 重新计算，仅作诊断，不拟合部署阈值。

所有区间都是**固定三个 checkpoints 下的 image-sampling 探索性95%区间**，不包含未来训练 seed 的不确定性。下面所有风险、效应及 AUROC 都乘100；AUGRC 原始范围为 [0,0.5]，不能将“×100点”写成相对改善百分比。G/S 为 Global-max／Native-selected，E/Y 为 exists／emit。

## 3. 四格主指标与绝对改善／交互

每格为 mixed AUGRC×100 的 **mean ± sample SD [95% CI]**，越低越好。六个 JSON block 同时保留完整逐 seed 值，不能用总体区间掩盖 seed 差异。

| Localizer／面 | G/E | G/Y | S/E | S/Y |
|---|---|---|---|---|
| MM／FineCops | 26.229±0.069 [25.844,26.603] | 26.378±0.039 [26.011,26.738] | 26.196±0.072 [25.808,26.574] | 26.355±0.044 [25.990,26.714] |
| MM／gRef Full | 28.790±0.079 [28.306,29.276] | 28.750±0.046 [28.263,29.234] | 28.765±0.110 [28.283,29.250] | 28.761±0.047 [28.275,29.246] |
| MM／gRef disjoint | 28.679±0.077 [28.127,29.214] | 28.612±0.041 [28.072,29.152] | 28.652±0.109 [28.100,29.191] | 28.635±0.047 [28.092,29.168] |
| MDETR／FineCops | 29.706±0.273 [29.296,30.115] | 28.841±0.127 [28.460,29.217] | 29.516±0.278 [29.102,29.921] | 28.652±0.017 [28.269,29.030] |
| MDETR／gRef Full | 20.249±0.535 [19.836,20.673] | 20.087±0.392 [19.687,20.516] | 20.516±0.139 [20.098,20.942] | 20.526±0.055 [20.120,20.945] |
| MDETR／gRef disjoint | 20.106±0.536 [19.655,20.575] | 19.917±0.384 [19.476,20.379] | 20.342±0.151 [19.891,20.805] | 20.354±0.065 [19.904,20.816] |

令 `D_emit = R(S,Y)−R(G,Y)`、`D_exists = R(S,E)−R(G,E)`、`I = D_emit−D_exists`。效应与其配对区间如下；与四格原始风险一起解释，不相互替代。

| Localizer／面 | D_emit [95% CI] | D_exists [95% CI] | I [95% CI] |
|---|---|---|---|
| MM／FineCops | −0.023042 [−0.048349,+0.002569] | −0.033883 [−0.067824,+0.000049] | +0.010841 [−0.016281,+0.037854] |
| MM／gRef Full | +0.011282 [−0.021118,+0.044859] | −0.025280 [−0.054632,+0.005792] | +0.036562 [+0.002771,+0.069016] |
| MM／gRef disjoint | +0.022305 [−0.012845,+0.057311] | −0.026568 [−0.060345,+0.007270] | +0.048873 [+0.013208,+0.084921] |
| MDETR／FineCops | −0.189079 [−0.256630,−0.122089] | −0.190574 [−0.285065,−0.093888] | +0.001495 [−0.090963,+0.087954] |
| MDETR／gRef Full | +0.439503 [+0.373097,+0.505851] | +0.267601 [+0.175659,+0.356319] | +0.171902 [+0.084867,+0.258765] |
| MDETR／gRef disjoint | +0.436960 [+0.363368,+0.506907] | +0.235657 [+0.137095,+0.331584] | +0.201303 [+0.108389,+0.291852] |

这支持三个有区分度的结论：

- MM：主要指标的 emit 改善未确认；gRef 的正 interaction 也不能称“修复 emit”。FineCops 中 S/Y−S/E = **+0.159465 [0.076733,0.241143]**，原目标差异在 matched Selected 下保留。
- MDETR／FineCops：Selected 明确改善两个目标，幅度接近；没有确认 emit 特异性改善。不能把 `D_emit<0` 与 `I<0` 混成同一个条件。
- MDETR／gRef：Selected 的 emit 退化在两个面、三个 seed 都为正；同时 C–W、C–N 均下降。例如 disjoint 的变化分别为 **−1.010339 [−1.581021,−0.441654]** 与 **−1.931599 [−2.215403,−1.638419]**。这是 matched-readout 的冻结迁移边界，不是缺少 Selected 训练。

**MDETR 的部分平均效应有明显 seed 异质性。** 以 disjoint 为例：

| 效应，×100 | seed17 | seed42 | seed73 | mean ± sample SD |
|---|---:|---:|---:|---:|
| G/Y−G/E | −1.032384 | +0.074417 | +0.391854 | −0.188704±0.747689 |
| D_emit | +0.872024 | +0.067664 | +0.371190 | +0.436960±0.406193 |
| I | +1.155431 | −0.215211 | −0.336311 | +0.201303±0.828515 |

不能将前两张表的 image-bootstrap 区间描述为所有训练 seed 都同方向，也不做 n=3 t-test。完整 `per_seed`、每项 `summary.sample_sd` 和效应 SD 均在源 JSON；这里只突出最容易被平均值掩盖的例子。

## 4. 三状态告诉我们风险为什么变，而不是只看 existence AUROC

C 为输出框正确，W 为正例但输出框错误，N 为 no-target。下表统一是 **Global emit−exists**，AUROC points；括号为95%区间。

| Localizer／面 | C–W | C–N | W–N | Existence AUROC | Mixed AUGRC×100 |
|---|---|---|---|---|---|
| MM／FineCops | +11.654 [10.721,12.641] | −3.523 [−3.884,−3.166] | −17.795 [−18.821,−16.781] | −6.754 [−7.211,−6.308] | +0.149 [0.058,0.235] |
| MM／gRef Full | +6.190 [5.306,7.049] | −3.166 [−3.569,−2.754] | −6.927 [−7.565,−6.281] | −4.823 [−5.234,−4.410] | −0.040 [−0.143,0.062] |
| MM／gRef disjoint | +6.533 [5.605,7.467] | −3.184 [−3.635,−2.751] | −7.204 [−7.895,−6.501] | −4.951 [−5.398,−4.511] | −0.066 [−0.178,0.043] |
| MDETR／FineCops | +13.889 [13.044,14.743] | +0.290 [−0.265,0.857] | −14.257 [−15.124,−13.406] | −4.719 [−5.271,−4.165] | −0.866 [−0.988,−0.745] |
| MDETR／gRef Full | +5.752 [4.601,6.891] | −0.521 [−0.896,−0.127] | −7.466 [−8.931,−6.040] | −1.783 [−2.206,−1.349] | −0.162 [−0.265,−0.065] |
| MDETR／gRef disjoint | +6.204 [5.153,7.325] | −0.493 [−0.917,−0.082] | −7.990 [−9.622,−6.478] | −1.844 [−2.322,−1.395] | −0.189 [−0.291,−0.083] |

两个 localizer、三个面、两种 matched readout 均保留 C–W 增益及 existence AUROC 下降的平均方向和相应区间；但 **C–N 不同，完整风险也不同**。MDETR FineCops 的 C–N 区间跨零，不能声称 C–N 已确认改善或精确不变。其 W–N 明显下降，说明 existence AUROC 混入了对二元成功排序不直接有用的两类失败之间比较。

设 `a=P(C|positive)`、`π=P(N)`，固定预测下：

```text
AUC_exists = a U_CN + (1−a) U_WN
ΔAUGRC = −(1−π)a[(1−π)(1−a)ΔU_CW + πΔU_CN]
```

FineCops 的 Global 点估计分解为：MM 的 C–W 项 **−0.532452** 与 C–N 项 **+0.681076** 相加得 **+0.148624**；MDETR 则为 **−0.818060** 与 **−0.047500**，相加得 **−0.865560**。这是精确风险恒等式的点估计分解，不为两个派生项另造区间，也不把 MDETR 的小 C–N 点收益升级成已确认机制。

在 MM／FineCops 的 nominal50% coverage，G/E 的 accepted W/N 比例为 **8.441699% / 41.457159%**，G/Y 为 **3.554400% / 46.810432%**：较少 wrong boxes 被更多 no-target 接受抵消。实际 coverage 为50.002709%；这里分母为全部 accepted records，**不是 no-target-class FPR**。其余覆盖率和配对区间均保留在 `*_risk_cov10/25/50/75/90/100`；没有用这些点拟合部署策略。

不能以辅助积分掩盖主指标：MM／disjoint 的 G/Y−G/E mixed AURC×100 为 **−0.397397 [−0.765277,−0.032938]**，而 AUGRC 的差异未分清。MDETR／FineCops 的 `D_emit` AUGRC 改善明确，但 AURC 为 **−0.039976 [−0.336518,+0.253362]**。因此“某一监督目标总是提高完整风险”或“两种积分总能认证同一优劣”都不是本结果支持的主张。

### 风险交点与缺失交点

MM／FineCops 的 Global／Selected crossover prior 分别为 **0.428197 [0.393234,0.463998]**、**0.419507 [0.385675,0.455559]**，均有5,000个内部交点；observed π=0.489244 在其上方。MM／disjoint 的 Global 为 **0.474239 [0.417016,0.532642]**，observed π=0.439308，净收益接近交点且区间尚未区分。

MDETR／FineCops 的两个平均目标对照**没有内部交点**；Global 的5,000轮中4,241轮也无内部交点，不能裁剪到1。MDETR／disjoint 的 Global 点交点0.694166，但45轮无内部交点：普通CI为null，只能报告其4,955个内部根的条件区间 **[0.532217,0.913746]** 并同时披露缺失数。这些是固定得分分布下的先验敏感性，不是 AURC 闭式公式或部署 prior 建议。

## 5. 条件化诊断和空间解释的边界

FineCops 的 negative parent 全为 L1。MM 的 Global C–N 总变化 −3.523202 中，within-level 贡献为 +0.169020、cross-level 为 −3.692222；MDETR 对应为 +1.626950／−1.337023，总变化 +0.289927。两者的 L1-only C–N 变化分别为 **+0.223845 [0.024417,0.431406]**、**+2.194837 [1.646792,2.750394]**。这把难度／监督覆盖接到了 C–N 风险，而不只重复 C–W 的大 AUROC 数字。L2/L3 没有同难度 negatives，相关条件AUROC保持未定义；不能换用 negative edit level。

同图分析也没有把目标效应解释掉：MM／FineCops 的 Global 同图 C–W **+7.642212 [5.849847,9.420778]**、C–N **−1.163453 [−1.557184,−0.759611]**；MDETR 对应为 **+11.751674 [10.280273,13.309221]**、**+1.307116 [0.508897,2.098384]**。C–N 可比面分别含2,344／2,194图、32,776／26,598图内pairs。同图／可比无条件的衰减及其配对区间见 JSON；MM C–W 衰减区间跨零，不构成“主要来自图像级排序”的证据。

真实 edited-negative/parent-positive 的9,029对也不能混淆状态：MDETR 的 Global all-parent win 下降 **−1.683464 [−2.561608,−0.822756]**，但 C-parent 上升 **+1.143571 [0.186539,2.067830]**，W-parent 下降 **−9.699802 [−11.789176,−7.615188]**。分别有6,675个C-parent pairs和2,354个W-parent pairs；“所有parent正例优于negative”的目标不同于“正确输出优于失败”。gRef 没有虚构这种编辑parent关系，只做 same-image。

固定权重的跨读数诊断同样具有方向差异：

| Localizer／面 | 读数切换 | Δ mixed AUGRC×100 [95% CI] |
|---|---|---|
| MM／FineCops | G/Y→Selected | +0.057874 [0.019445,0.097048] |
| MM／FineCops | S/Y→Global | +2.029083 [1.875547,2.178650] |
| MDETR／FineCops | G/Y→Selected | +1.607516 [1.513385,1.697795] |
| MDETR／FineCops | S/Y→Global | +0.435035 [0.371218,0.494901] |
| MM／gRef disjoint | S/Y→Global | +1.865768 [1.671612,2.058596] |
| MDETR／gRef disjoint | S/Y→Global | −0.109195 [−0.167168,−0.049037] |

所以，读出不是可以任意互换的实现细节，但也不能说“失配必定有害”。最后一行是真实的反例。跨读数与重新训练是不同干预；G→S 同时改变评分query、query竞争和训练梯度落点。已保存的 winner index、box IoU、GT IoU、C/W/N 不一致率可描述这些变化，不能唯一分配“空间因素贡献率”。**三状态恒等式解释风险由哪些排序项构成，并没有单独识别优化为何学出这些排序项。**

## 6. 两个固定组合：必须同时报告三个参照

以下均为组合的 mixed AUGRC×100 减去对应参照；负数为改善，区间是配对 image bootstrap。组合使用同 seed 的 G-exists；Product 无拟合参数，SIRC-style 的均值／总体SD仅来自83,341条唯一训练positive，不根据评价结果选权重或阈值。

| Localizer／面／组合 | 相对 Native | 相对 G/E | 相对 G/Y |
|---|---|---|---|
| MM／FineCops／Product | −1.497 [−1.598,−1.398] | +0.386 [0.286,0.488] | +0.238 [0.177,0.302] |
| MM／FineCops／SIRC-style | −0.758 [−0.821,−0.696] | +1.125 [0.972,1.280] | +0.977 [0.865,1.090] |
| MM／gRef Full／Product | −1.629 [−1.771,−1.484] | −0.175 [−0.253,−0.098] | −0.134 [−0.189,−0.080] |
| MM／gRef Full／SIRC-style | −1.382 [−1.507,−1.252] | +0.073 [−0.035,0.184] | +0.113 [0.054,0.176] |
| MM／gRef disjoint／Product | −1.569 [−1.727,−1.407] | −0.196 [−0.282,−0.111] | −0.130 [−0.187,−0.074] |
| MM／gRef disjoint／SIRC-style | −1.338 [−1.482,−1.191] | +0.035 [−0.088,0.157] | +0.101 [0.040,0.164] |
| MDETR／FineCops／Product | −0.831 [−0.999,−0.663] | −0.381 [−0.439,−0.321] | +0.485 [0.385,0.589] |
| MDETR／FineCops／SIRC-style | −0.146 [−0.159,−0.134] | +0.304 [0.107,0.493] | +1.170 [1.038,1.302] |
| MDETR／gRef Full／Product | −1.283 [−1.465,−1.102] | −0.261 [−0.288,−0.234] | −0.099 [−0.187,−0.008] |
| MDETR／gRef Full／SIRC-style | −0.184 [−0.202,−0.161] | +0.839 [0.652,1.022] | +1.001 [0.862,1.139] |
| MDETR／gRef disjoint／Product | −1.291 [−1.485,−1.088] | −0.265 [−0.295,−0.237] | −0.076 [−0.170,+0.015] |
| MDETR／gRef disjoint／SIRC-style | −0.184 [−0.204,−0.156] | +0.843 [0.644,1.039] | +1.031 [0.881,1.179] |

MM 的 Product transfer 是可保留的实践结果，不应沿用“两个组合都失败”的阶段性措辞。MDETR 的 Product 在 disjoint 相对 G/Y 仍未分清，不能把 Full 的较小优势无条件推广；SIRC-style 优于 Native 也不等于优于已学头。所有改善只针对本表的人群、指标与固定得分，不能写成已校准的成功概率或无需验证的部署政策。

## 7. 论文应围绕什么组织

主线应是：**固定输出的标签干预改变失败排序 → matched读出检验其结构边界 → C/W/N解释完整风险 → 第二模型与冻结迁移显示适用范围 → 固定组合提供有参照、有代价的实践启示。**

可以保留的明确知识增量是：在固定 grounder／框／头结构／训练事件下，改变 W 的监督标签，能够受控地改变 C–W 和无目标相关排序；该目标效应不等于单一读出失配，且 existence AUROC 不能认证完整输出排序。不能继续主张普遍监督冲突、Selected空间修复、普遍不可以组合或某个目标跨所有 seed 必胜。

L1-only 监督覆盖是重要限定和后续可检验解释，不是这次结果已识别的唯一根因。当前研究没有通过重新平衡或追加L2/L3训练来选择一个更好结果。Record-only 工具是可复用交付物，不与核心受控发现并列包装为一种新理论。

## 8. 完成与复现绑定

三份实际输出均为完整两个 localizer、三 seed、全部5,000draws，原三条统计进程未被并行替代工具中断。FineCops 完整结果的 MM block 与先前 staged block **dict-wise完全相同**。模型、分数和实验结论未因 metadata-v2 的 unused edit-level 修正而改变；具体兼容性处理见[执行 ledger](confidence_readout_v6_20260906.md)。

| 结果 | SHA-256 |
|---|---|
| [FineCops val](../paper/data/readout_v6/finecops_val.json) | `91a088208d0a2786f9a07de2d9248ffb0a273c0e78c32f5de7990e4acc03196d` |
| [gRef Full](../paper/data/readout_v6/gref_full.json) | `7d8c2b9db2fdf90b9e9b9218383afef4a51b17089763ca85f87742963b33c5f9` |
| [gRef source-disjoint](../paper/data/readout_v6/gref_finecops_train_val_source_disjoint.json) | `33cbd5765b6cf06dcc11fb21545aeb0456f839d0c5c586baf47122baccb674fd` |
| [实验完成收据](../paper/data/readout_v6/experimental_completion.json) | `8ea84eb11c122c02b9b95e4d890af7d9f33a4ab9c6b702a40ba12b3971aac415` |

协议 SHA：`bc39843bd4694d80dee3e623bcb40f05f30f9df92083d5a3937dfcf1b4093a1e`。[正常退出记录](../paper/data/readout_v6/completion_terminal.json)为returncode0；实验收据记录18个新头、0 trunk updates、0新FineCops Test forwards、无checkpoint selection／threshold fitting。每份 JSON 的 `receipt` 还绑定数值代码、逐记录结果与六份训练 SIRC 统计的SHA；`localizers.*.per_seed/summary/effects/conditional_counts/winner_geometry` 保留本文没有逐项展开的完整结果。

CPU-only复现入口、输入schema、路径迁移方法和失败边界见[复现指南](readout_v6_reproduction.md)。这些完成收据不代替论文逐页检查或公开发布审计。
