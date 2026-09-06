# v6 第一模型阶段结果：目标效应在 Selected 读出下仍存在，但没有确认目标特异性修复

状态：**FIRST-LOCALIZER STAGED ANALYSIS — 不是完整 v6 结论，不替换封存 v5。**

唯一结果来源：[finecops_val_mm_stage.json](../paper/data/readout_v6/finecops_val_mm_stage.json)。SHA-256：`7742757dc391f9a4833fd2988b78ad6c59f9ba03c89b64c6721b68f12780165e`。本文件只读取这份已完成的结果，未训练、未修改模型、未重算或选择 checkpoint，未读取 MDETR／gRef 新结果。

## 1. 现在可以说什么

在这个冻结 FineCops positive-source MM-GDINO-T、匹配训练配方和三 seed 的设置中，**直接监督正确输出带来的定位排序增益与 C–N 排序损失，在 Selected-query 读出下依然出现**。因此，不能把原现象完全归结为“global-max 读取了另一个 query”，也没有证据说明换成 Selected 就修复了 emit 的完整输出风险。

这不是“读出毫无作用”：固定权重后违背训练时的读出方式，尤其把 Selected-trained 头改成 global-max，会明显损害完整输出排序。更准确的实践结论是：**读出方式是训练／部署契约的一部分；对齐读出并不自动消除监督目标带来的排序取舍。** 空间对应关系依然只是待解释因素之一，不是已识别的唯一原因。

## 2. 口径、四格结果和绝对效应

评估面为 FineCops val：18,455 records、3,567 image clusters；9,426 positive 中 C=7,292、W=2,134，N=9,029。Native P@1 固定为 **77.360492%**。自然 no-target 比例为 `9029/18455 = 0.4892441073`。

seeds 为 17/42/73，5,000 次 paired image-cluster bootstrap，PCG64 seed `20260911`。所有头和三个 seed 共用 image draws；先逐 seed 计算指标，再等权平均。区间描述固定 checkpoints 下的 image-sampling 不确定性，不是对未来训练 seed 的总体推断。每轮诊断 FPR95 重新计算 positive q05，不拟合部署阈值。

下文 AUROC／比例均乘100；**AUGRC、AURC及其差值也统一乘100**，简称“×100点”，不能当成相对百分比改善。原始 AUGRC 范围为 [0,0.5]。

| 匹配训练／部署路由 | Mixed AUGRC×100，mean ± sample SD | 95% paired cluster CI | Mixed AURC×100 |
|---|---:|---:|---:|
| G-exists | 26.229487 ± 0.069038 | [25.844024, 26.603484] | 47.388327 |
| G-emit | 26.378111 ± 0.039084 | [26.010877, 26.737886] | 47.685337 |
| S-exists | 26.195604 ± 0.072260 | [25.807766, 26.574376] | 47.256541 |
| S-emit | 26.355069 ± 0.043639 | [25.989956, 26.714256] | 47.580809 |

令 R 为 mixed AUGRC，G/S 分别表示 global-max／Native-selected：

| 主要效应 | 差值×100 | 95% CI |
|---|---:|---:|
| D_emit = R(S,emit) − R(G,emit) | −0.023042 | [−0.048349, +0.002569] |
| D_exists = R(S,exists) − R(G,exists) | −0.033883 | [−0.067824, +0.000049] |
| I = D_emit − D_exists | +0.010841 | [−0.016281, +0.037854] |
| G 内 emit − exists | +0.148624 | [+0.058134, +0.235490] |
| S 内 emit − exists | +0.159465 | [+0.076733, +0.241143] |

`D_emit` 原始尺度为 −0.0002304208，I 为 +0.0001084055。Selected 的 emit 点估计略好，但主要指标区间尚不能确认改善；interaction 的点估计也不是预期的负方向。相较约0.15×100点的目标差异，读出效应较小，两种读出内的 emit−exists 风险差异均保持为正。不能据此宣称精确等效或“Selected 没有任何效果”。

逐 seed 的主要结果同样必须保留：

| Seed | G-exists | G-emit | S-exists | S-emit | D_emit | D_exists | I |
|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | 26.174666 | 26.356751 | 26.121690 | 26.315697 | −0.041053 | −0.052976 | +0.011923 |
| 42 | 26.307019 | 26.423220 | 26.266087 | 26.401991 | −0.021229 | −0.040932 | +0.019702 |
| 73 | 26.206775 | 26.354362 | 26.199035 | 26.347519 | −0.006843 | −0.007740 | +0.000896 |

AURC 不能隐去：D_emit 的 mixed AURC×100 为 **−0.104528 [−0.182868, −0.025074]**，D_exists 为 −0.131786 [−0.222306, −0.042495]，interaction 为 +0.027258 [−0.046797, +0.102632]。因此 AURC 确认较小的一般性读出改善，但同样没有确认 emit 特异性改善；S 内 emit−exists 仍为 +0.324267 [+0.151384, +0.491331]。不能用这一辅助指标代替预先规定的 AUGRC 主指标。

## 3. 三状态排序解释完整风险，而不混淆存在性 AUROC

下表为 emit−exists，AUROC points：

| 状态对／指标 | Global-max 差值 [95% CI] | Selected 差值 [95% CI] |
|---|---:|---:|
| C–W：正确框高于错误框 | +11.653804 [10.720949, 12.640734] | +10.723642 [9.850165, 11.606102] |
| C–N：正确框高于无目标 | −3.523202 [−3.883794, −3.165805] | −3.359437 [−3.699552, −3.018957] |
| W–N：错误正例高于无目标 | −17.794869 [−18.820577, −16.781050] | −16.823466 [−17.738263, −15.898368] |
| Existence AUROC | −6.754237 [−7.211445, −6.307960] | −6.407627 [−6.827402, −5.984316] |

所以，存在性 AUROC 下降不等于同等幅度的成功排序损失。设 a=P(C|positive)=0.7736049226：

`AUC_exists = a U_CN + (1−a) U_WN`。

例如 Global 的 −6.754237 点中，C–N 项贡献 −2.725567，W–N 项贡献 −4.028671。后者区分的是两类失败，**并不直接进入“成功 vs 失败”的二元完整输出风险**。

完整输出风险的精确分解为：

`ΔAUGRC = −(1−π)a[(1−π)(1−a)ΔU_CW + πΔU_CN]`。

代入 observed π，得到以下**点估计代数贡献，不为这些派生项另造区间**：

| emit−exists | C–W 改善对 AUGRC×100 的贡献 | C–N 损失的贡献 | 合计 |
|---|---:|---:|---:|
| G | −0.532452 | +0.681076 | +0.148624 |
| S | −0.489954 | +0.649419 | +0.159465 |

这回答了“较大的定位 AUROC 增益为何没有变成更好的完整输出排序”：在当前 population 中，C–N 的损失权重大于 C–W 的收益。不是因为 existence AUROC 的所有损失都同等重要，更不是定位框发生了变化。

Selected 相对 Global 的 emit 本身，也没有单向提升两个相关判别：C–W **−0.599886 [−0.829257, −0.371692]**，C–N **+0.260979 [+0.154131, +0.366170]**。它重新分配了两种排序能力，净 AUGRC 改善很小；positive-only AURC×100 反而增加 +0.210141 [+0.134204, +0.286474]。

风险 crossover 是**请求比例交点，不是 confidence 阈值**：

| emit−exists 风险曲线 | 等 seed 平均风险差的内部交点 π* | 95% CI | Bootstrap内部根 |
|---|---:|---:|---:|
| G | 0.428196949 | [0.393233884, 0.463997720] | 5,000/5,000 |
| S | 0.419507310 | [0.385675187, 0.455558819] | 5,000/5,000 |

Observed π=0.489244 高于两个交点及其区间上界，与自然混合下 exists 的较低 AUGRC 一致。交点只适用于这些固定得分的 class-conditional distributions 和错误定义；不据此选择部署 prior／threshold。

## 4. 读出契约确实重要，但并未识别唯一的空间原因

固定每个已训练头，只改推理读数；下表为“改读数 − 匹配部署”的 mixed AUGRC×100：

| 固定训练头与推理切换 | ΔAUGRC×100 [95% CI] |
|---|---:|
| G-exists → selected | +0.015831 [−0.030490, +0.060974] |
| G-emit → selected | +0.057874 [+0.019445, +0.097048] |
| S-exists → global-max | +1.506023 [+1.364696, +1.644990] |
| S-emit → global-max | +2.029083 [+1.875547, +2.178650] |

S-trained 头改读 global-max 时，C–N 分别下降 **−7.974934 [−8.593114, −7.355105]**、**−8.987930 [−9.675902, −8.297233]** AUROC points。相比之下，G-trained 头切换 selected 的损失小得多。该非对称性支持“不能把未经相同训练聚合约束的全部 query maxima 当成可互换的部署 confidence”。它不能单独说明损失中有多少来自空间对应、max竞争或训练梯度落点。

以下所有 `winner` 都指该头 **dense logits 的 global-max query**。对 G 路由，它是实际 confidence 来源；对 S 路由，它只是反事实诊断。**S 的匹配部署始终读取 Native-selected query，其实际读出不一致率是零**，不能把表中 S 的比例误写成部署读错 query。

| 训练头 / seed | Global winner≠Native：C / W / N (%) | Winner box与Native box平均IoU：C / W / N | W中global winner反而正确 (%) |
|---|---:|---:|---:|
| G-exists / 17 | 7.830 / 29.288 / 12.504 | 0.9406 / 0.7261 / 0.9110 | 14.058 |
| G-exists / 42 | 7.652 / 28.772 / 8.761 | 0.9419 / 0.7312 / 0.9403 | 13.355 |
| G-exists / 73 | 8.338 / 31.771 / 9.946 | 0.9357 / 0.7030 / 0.9324 | 15.136 |
| G-emit / 17 | 4.841 / 18.322 / 5.826 | 0.9673 / 0.8302 / 0.9670 | 9.044 |
| G-emit / 42 | 4.676 / 17.245 / 5.250 | 0.9682 / 0.8400 / 0.9696 | 8.435 |
| G-emit / 73 | 5.047 / 19.025 / 8.882 | 0.9658 / 0.8273 / 0.9472 | 8.997 |
| S-exists / 17 | 35.793 / 58.950 / 55.997 | 0.8694 / 0.5895 / 0.8124 | 17.760 |
| S-exists / 42 | 61.766 / 76.757 / 77.484 | 0.8310 / 0.5421 / 0.7915 | 15.230 |
| S-exists / 73 | 46.558 / 64.292 / 66.320 | 0.8266 / 0.5314 / 0.7636 | 16.776 |
| S-emit / 17 | 58.475 / 85.333 / 70.351 | 0.8663 / 0.6057 / 0.8205 | 17.104 |
| S-emit / 42 | 58.653 / 84.067 / 71.636 | 0.8720 / 0.6392 / 0.8336 | 17.010 |
| S-emit / 73 | 62.685 / 85.239 / 74.183 | 0.8536 / 0.5787 / 0.8126 | 17.994 |

这些是逐 seed 描述统计，不冒充 bootstrap geometry 区间。不同 query 经常仍预测重叠框，因此索引不一致也不等于类别或空间语义完全不一致。Global emit 的 winner 已比 Global exists 更常对应 Native，却仍没有更低 mixed AUGRC；这进一步限制了简单的“只要 query 对应就修复风险”叙述。

## 5. 同图、真实 parent-pair 和难度：哪里保留，哪里衰减

### 5.1 同图比较不是 expression IID，也不是全人群的直接替代

同图 AUROC 按图内有效状态对数加权；bootstrap抽到某图 m 次，就携带 m 份该图的内部对，而不是把内部对数乘 m²。可比 unconditional 指标只使用同一批有效图上的同一组状态记录，允许跨图配对。

| 状态对 | 有效images | 图内pairs | 可比records | G：可比unconditional / 同图效应 | S：可比unconditional / 同图效应 |
|---|---:|---:|---:|---:|---:|
| C–W | 835 | 5,016 | 3,943 | +8.853561 / +7.642212 | +8.054308 / +7.535885 |
| C–N | 2,344 | 32,776 | 14,749 | −1.700259 / −1.163453 | −1.574731 / −0.933610 |
| W–N | 624 | 6,110 | 4,040 | −12.752597 / −10.643753 | −12.168241 / −9.989089 |

所有效应仍指 emit−exists。关键区间：

- 同图 C–W：G +7.642212 [5.849847, 9.420778]；S +7.535885 [5.863637, 9.221980]。收益在同图比较中仍然存在。
- C–W 的“可比unconditional−同图”效应差：G +1.211350 [−0.221634, +2.633502]；S +0.518423 [−0.843770, +1.911456]。不能据这些跨零区间称其主要是图像级排序。
- 同图 C–N：G −1.163453 [−1.557184, −0.759611]；S −0.933610 [−1.331714, −0.523227]。关键成功／无目标损失在同图内也未消失。
- C–N 的signed“可比unconditional−同图”差：G −0.536806 [−0.876540, −0.202394]；S −0.641121 [−0.978234, −0.304731]，显示负效应幅度有所衰减，而不是被完全解释掉。
- 同图 W–N：G −10.643753 [−12.590168, −8.700417]；S −9.989089 [−11.974956, −8.090608]。

### 5.2 真实 edited-negative／parent-positive 配对

总计9,029 pairs、2,433 images；其中 C-parent 为8,136 pairs／2,303 images，W-parent 为893 pairs／348 images。pair数不是unique parent数；两个parent状态面的image数也不能相加当作互斥图像数。这是固定得分的 pair-win 诊断，不重命名为官方 Recall@1。

| Parent面 | G-exists / G-emit win率 (%) | G emit−exists [95% CI] | S-exists / S-emit win率 (%) | S emit−exists [95% CI] |
|---|---:|---:|---:|---:|
| 全部 | 67.091225 / 66.596522 | −0.494702 [−0.986021, −0.026942] | 67.065382 / 66.423007 | −0.642375 [−1.114490, −0.163822] |
| C-parent | 67.985906 / 67.756473 | −0.229433 [−0.650801, +0.208362] | 67.985906 / 67.522943 | −0.462963 [−0.895500, −0.016347] |
| W-parent | 58.939903 / 56.028369 | −2.911534 [−5.904802, 0.000000] | 58.678611 / 56.401642 | −2.276969 [−4.812482, +0.219631] |

W-parent 的更大负点估计仍有较宽区间；不能据其大小断言该子群承担了全部机制。Global C-parent 也尚未分清，而不是已证明无效应。

### 5.3 最值得继续写清楚的 composition 线索

正例难度分布为 L1：C=5,506/W=591；L2：C=1,566/W=1,318；L3：C=220/W=225。**全部9,029 negative text 的原始 parent positive 都是 L1。** 因此 C–N／W–N 的“同难度”只存在 L1 面；不能伪造 L2/L3 negatives，不能将 negative edit level 代替正表达难度。

| emit−exists 分解，AUROC points | G [95% CI] | S [95% CI] |
|---|---:|---:|
| C–W：L1内 | +7.084626 [6.022783, 8.236068] | +6.725852 [5.725543, 7.768567] |
| C–W：L2内 | +6.689041 [4.880354, 8.599760] | +4.923882 [3.275290, 6.674540] |
| C–W：L3内 | +7.931987 [3.335004, 12.647208] | +7.647138 [3.467583, 11.743446] |
| C–W：同层pair加权AUROC增益 | +6.940326 [5.992502, 7.944031] | +6.041434 [5.167064, 6.965249] |
| C–W：within-level对总增益的贡献 | +2.393942 [2.056728, 2.752446] | +2.083885 [1.764950, 2.416818] |
| C–W：cross-level对总增益的贡献 | +9.259862 [8.506250, 10.049779] | +8.639757 [7.938139, 9.345548] |
| C–N：L1内AUROC变化 | +0.223845 [0.024417, 0.431406] | +0.331884 [0.144152, 0.526315] |
| C–N：within-level对总变化的贡献 | +0.169020 [0.018574, 0.325809] | +0.250597 [0.109324, 0.396528] |
| C–N：cross-level对总变化的贡献 | −3.692222 [−4.018634, −3.383512] | −3.610034 [−3.924596, −3.315666] |

C–W 的同层pair占34.493219% [32.989460%, 36.074726%]，余下为跨层pair。Global 总增益的79.457851%、Selected的80.567377%来自cross-level**代数贡献**。这不是“因果解释比例”：每个难度层内仍有清晰正收益，且跨层pair本来就占多数。

更直接相关的发现是：**C–N 的总体损失集中在 L2/L3 正确输出与 L1-parent negatives 的跨层比较；L1 C–N 内的变化反而略正。** 这把难度分析接到了完整风险损失上，而不只展示11.65点定位增益的来源。它是当前数据composition下的明确描述性线索，尚不能证明 head 显式估计了难度，或难度是唯一因果因素；同图 C–N 的损失也提醒我们不能退回“全部只是不同图像”的解释。

## 6. 固定组合：优于 Native，不等于优于已学头

两种组合均使用相同 seed 的 G-exists；只用全部83,341条唯一训练正例拟合 SIRC-style 的均值／总体标准差，不调 val 权重或阈值。

| 参照／组合 | Mixed AUGRC×100 | Mixed AURC×100 |
|---|---:|---:|
| Native | 28.112745 | 54.570520 |
| G-exists | 26.229487 | 47.388327 |
| G-emit | 26.378111 | 47.685337 |
| 固定 product | 26.615865 | 48.896006 |
| 固定 SIRC-style | 27.354697 | 53.103376 |

| ΔMixed AUGRC×100 | Product [95% CI] | SIRC-style [95% CI] |
|---|---:|---:|
| 组合 − Native | −1.496881 [−1.597938, −1.398160] | −0.758049 [−0.820664, −0.696297] |
| 组合 − G-exists | +0.386378 [+0.285510, +0.488328] | +1.125210 [+0.971852, +1.280392] |
| 组合 − G-emit | +0.237754 [+0.176863, +0.301855] | +0.976586 [+0.865084, +1.090404] |

它们在当前mixed指标下优于Native，但同时劣于两个G头；不能只选Native参照写“组合解决了取舍”。与此同时，product／SIRC-style相对G-exists改善positive-only AURC×100（−4.112675／−4.538668），相对G-emit也改善该指标（−0.541720／−0.967713），代价是更差的C–N与mixed风险。

可陈述“这些固定组合改变了能力分配，且相对Native有收益”；不能据它们未超过G头推导两种能力不可组合，也不把SIRC-style的这一实例当作对完整SIRC方法的普遍否定。

## 7. 对主线的阶段性影响

可以保留、并比只谈两个头更具体的证据链：

1. 在固定框和匹配head下，监督目标改变C–W／C–N排序；换成Selected后，方向和自然混合下的目标差距仍存在。
2. 三状态风险权重精确解释为什么当前population下C–N损失压过C–W收益；W–N只解释部分existence AUROC变化，不能直接充当完整风险原因。
3. 难度composition把C–N损失定位到具体比较面，同时同层／同图结果阻止我们把它简单归因为图像或难度先验。
4. 固定权重跨读数表明训练／部署聚合契约重要，但未识别唯一空间归因；两种固定组合提供受限的实践对照，而非解决方案胜利。

仍未解决：独立localizer上的范围、冻结迁移、何种表示或估计误差使直接成功监督未形成更好的总体成功排序。**不根据本阶段结果改变后续MDETR矩阵、head配方或既定评估面；不提前生成完整 v6 或跨benchmark结论。**

## 证据绑定

- Study protocol SHA：`bc39843bd4694d80dee3e623bcb40f05f30f9df92083d5a3937dfcf1b4093a1e`。
- Staged analysis input SHA：`38b182c3631b513ac1b0acc05906367e51d39e0460ffad9f58efe9150735ad3b`。
- 共享draw SHA：`b453e3a9081de403d1a7a92a26699163446244ef9fba24cfeece24c338c07167`。
- 冻结分析core SHA：`17c922f833c29155b01cdd870a2d4746b7fee7c5fc13c99773253a60e22ca946`；CLI SHA：`85bdb6cc2199b1ea424742aed83534e7650df16ec2e7df6fea8a4a1e0f109774`。
- 原receipt明确 `stage_mm_only=true`、`formal_requested_configuration=false`、`study_final_receipt=false`。这里的 false 是未完成整个两localizer研究，并非本阶段bootstrap未跑满；本阶段iterations确为5,000。

## 8. 后续 metadata 核查：confidence 的直接监督只覆盖 L1 parent 正例

这项补充只读取并校验了封存 TRAIN／val 的 source annotation 与 cache index，未读取新模型结果、未更新 head、未重平衡数据。其结果进一步限制第1–7节的解释：**这里不是一个在全部难度正例上充分监督后、再评价同一正例人群的 confidence 实验。**

### 8.1 TRAIN：83,341 个可用正例不等于 83,341 个受监督正例

| 官方 positive difficulty | TRAIN 全部 unique 正例 | 进入 confidence pairs 的 unique parent 正例 | 每 epoch 的 pair-weighted 正例次数 | 未进入 head loss 的 unique 正例 | unique 正例覆盖率 |
|---|---:|---:|---:|---:|---:|
| L1 | 54,015 | 43,979 | 80,451 | 10,036 | 81.419976% |
| L2 | 25,282 | 0 | 0 | 25,282 | 0% |
| L3 | 4,044 | 0 | 0 | 4,044 | 0% |
| 合计 | 83,341 | 43,979 | 80,451 | 39,362 | 52.769945% |

80,451 个 negative-text rows 对应80,451个same-image pairs，但只有43,979个不同的positive parents，分布于21,799张图。每个unique parent平均出现 `80451/43979 = 1.829304896` 次／epoch；全部pairs每个epoch都遍历，五个epoch因此是每头402,255次paired-positive呈现，以及402,255次negative呈现，不能当成这么多独立样本。

实际parent重复次数分布：

| 每个parent的negative edits数 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN unique parents | 21,729 | 12,575 | 6,110 | 2,778 | 623 | 133 | 31 |
| val unique parents | 2,437 | 1,434 | 674 | 321 | 65 | 12 | 3 |

TRAIN全部可用正例来自31,795张图；按L1/L2/L3分别覆盖24,338／11,698／2,499张图。难度图像集合可以重合，这三个数不能相加当作独立图像总数。

### 8.2 val：同样只有 L1 parents 构成配对面，但完整评价含 L2/L3

| 官方 positive difficulty | val 全部 unique 正例 | 具有negative pair的unique parents | pair-weighted parent次数 | 未配对正例 |
|---|---:|---:|---:|---:|
| L1 | 6,097 | 4,946 | 9,029 | 1,151 |
| L2 | 2,884 | 0 | 0 | 2,884 |
| L3 | 445 | 0 | 0 | 445 |
| 合计 | 9,426 | 4,946 | 9,029 | 4,480 |

L1配对覆盖率为81.121863%，全positive覆盖率为52.471886%；平均每个配对parent对应1.825515568个negative edits。全量positive评价仍包含3,329个L2/L3正例，占35.317208%，但 head 正例监督中对应难度的样本数为零。

完整TRAIN正例中的L2/L3比例其实是35.187963%，与val全正例接近。因此重点不是“完整FineCops TRAIN碰巧几乎没有难例”，而是**paired-only head loader将优化正例人群限制到了L1 parents**。

### 8.3 从代码确认的是 optimizer 的输入，而不只是 metadata 的存在

以下冻结路径共同构成证据：

- [`_validate_population`](../tools/train_finecops_bce_l2_heads.py#L307) 在321–336行先保留全部positive供rank列表使用，再为每条negative按 `parent_positive_id` 查找positive，构造pairs。pairs不是从全部positive均匀采样。
- [`_epoch_schedule`](../tools/train_finecops_bce_l2_heads.py#L339) 在343–358行生成全部positive的rank permutation和全部pairs的confidence permutation。生成rank事件只保留随机数／顺序契约，不意味着rank事件实际训练了head。
- 新版 [`train_confidence_readout_heads.py`](../tools/train_confidence_readout_heads.py#L261) 在261–274行跳过所有非confidence事件，随后只用 `batch = pairs[indices]`、`rows = pos + neg` 进入head loss。
- 旧封存Global头的 [`run_finecops_fixed_rank_targets.py`](../tools/run_finecops_fixed_rank_targets.py#L188) 在188–203行执行相同逻辑：rank事件生成后跳过，只对 `pos + neg` 计算exists／emit损失。因此这项人群边界同时适用于本阶段对照的G和S，不是新增Selected改变了数据。
- 全部83,341正例仍会被加载到cache、用于Native correctness标签审计；训练SIRC-style统计时，[evaluator](../tools/evaluate_confidence_readout_cache.py#L196) 也确实forward全部unique训练正例。**标签计算、缓存存在、无梯度统计forward，不等于这些样本进入了confidence优化。** 组合的统计人群比head实际监督人群更宽，也应披露。

不能把 `71,457/83,341` 的全TRAIN Native-correct审计比例，当作emit训练的正类比例；后者需要在80,451个pairs的实际parent重复权重下计数。本次metadata核查没有额外计算模型标签分布。

也不能写成“模型从未见过L2/L3”：这里限定的是 **confidence head 的直接positive-target loss**。已封存的 MM-GDINO positive-source trunk 配方使用全部83,341个正例，包含L2/L3；冻结表示因此可能包含这些表达的信息。该区分与[封存trunk recipe](../paper/tables/supp_positive_trunk_recipe_v5.tex)一致，本次没有重新审计或训练trunk。

此外，L1/L2/L3是官方positive difficulty标签，不等于语言结构复杂度。原negative annotation中可以出现 `2_hop`／`and` 表达，即使其 `positive_id` 对应L1 parent；不能将本次结果泛化为“head从未接触关系表达”。

### 8.4 对阶段性科学解释的修正

这是一条具体的 **head监督覆盖与完整评价人群之间的边界**：训练损失只覆盖paired L1 parents，完整评价则包含L1/L2/L3正确与错误输出。它与第5.3节的发现相吻合——C–N总体损失集中在L2/L3正确输出与L1-parent negatives的比较，而L1 C–N变化略正。

因此，不能把当前结果描述为“在同一完整目标分布上充分监督正确输出，仍原则上不能学到更好的成功排序”，或把它当作对理想 `P(Y=1|x)` 排序性质的反例。更准确的表述是：

> 在由paired L1正例监督、对完整难度人群评价的冻结grounder设置中，监督事件改变了C–W和C–N的排序；Selected读出并未消除这一现象。实际监督覆盖与评估composition是解释结果时必须显式保留的边界。

这一metadata发现不推翻G/S及exists/emit之间的匹配比较：四格仍共享同样的人群、更新顺序与参数结构；它改变的是现象的适用范围和理论解释。它也尚未证明这是主要因果根源：没有做改变head监督覆盖的受控干预，现有同图／难度分解仍属描述性分析。**不据此改动已经锁定的矩阵或追加重平衡训练，不提前推断第二localizer的结果。**

### 8.5 本次核查的精确来源

均在远端 `/mnt/why/PIVOT` 下读取；源文件保持原样。每个cache index row的 `level` 均与对应source annotation一致，每个negative的 `parent_positive_id` 均与官方annotation的 `positive_id` 一致，所有pairs均同图，TRAIN／val检查全部通过。

| 源文件 | SHA-256 |
|---|---|
| `data/FineCops-Ref/v1/raw/article_files/expression_all_train_set_coco_format.json` | `496f4152eddc5f1bedfb9f901d2f02e61ac2c2b0e10394f365f7c2a7df6cd9f8` |
| `outputs/b32a1_finecops_positive_trunk_factorized_20260904/formal_cache/train/manifest.json` | `23c1e61a33659fddbc153df1f7a13650d110df8c3caa3a17a0f5c2ab238120f0` |
| `outputs/b32a1_finecops_positive_trunk_factorized_20260904/formal_cache/train/index.jsonl` | `e9ced1b91b78735fe5fd3449c41fb5aafda9f809502e4541e66951dfa7ae71bf` |
| `data/FineCops-Ref/v1/raw/article_files/expression_all_val_set_coco_format.json` | `ac0bd7e9b883c100001c5bd202fc9dc08c038e79e49d2bb1755836a43d97f832` |
| `outputs/b32a1_finecops_positive_trunk_factorized_20260904/formal_cache/val/manifest.json` | `c3d30772ed048730b228a35e217e128c17bc3935e2bccb274ec69cbd4ddec085` |
| `outputs/b32a1_finecops_positive_trunk_factorized_20260904/formal_cache/val/index.jsonl` | `03cb08398b8d9aaa2a4127e90f29939f26c8a36356ea42682c1b1e5d67c8e96b` |

代码绑定：

- `tools/train_finecops_bce_l2_heads.py`：`0ad982801e8e1af096657cb006c57965ea51672ee268c634d80c4cfe11c0c5fd`。
- `tools/train_confidence_readout_heads.py`：`8a3bb55eb45b5e482db295ca6050b84e970215b2a43f3dd97ecdfc4ec4fbcfc3`。
- `tools/run_finecops_fixed_rank_targets.py`：`291f512bef731f7aacef4b8896899caff2d856f4dc9e09c84f9888dabc196783`。

前两者与本study protocol的冻结code bindings逐字节一致；没有改写任何训练／统计实现。
