# v8：将完成的覆盖干预写入论文主线

三套 5,000 次统计均已完成，退出码为0；12个头、三个seed、两次缓存评分及Native逐记录parity已通过。数据和训练保持封存，本轮只整合结果与重写论文，不新增训练、模型评分、bootstrap或FineCops Test访问。

当前论文：**Confidence for Which Prediction? Supervision Coverage Shapes Grounding Reliability**。

## 中心发现

监督事件不是脱离训练人群而独立生效的设计选择。保持grounder、输出框、Global-max读出、head结构、初始化、负例与更新预算固定，从全部L1均匀采样扩展为全正例均匀采样，使Y改善、E恶化，改变了两种监督的相对优劣。三状态比较说明这种变化来自哪些排序能力的交换，而不是把完整风险下降误写成所有正确性能力都恢复。

主文围绕一条论证组织：

**监督事件为什么不足以决定可靠性 → 读出对照排除简单修复 → 三状态比较定位损失 → 覆盖干预改变取舍 → 冻结迁移检验延续 → 固定组合提供实践补充。**

## 核心数值

以下全部为百分尺度的AUGRC点数，越低越好；区间是固定三seed平均的paired image-cluster 95%区间。

| 评价面 | Full−L1：Y | Full−L1：E | 交互 DY−DE |
| --- | --- | --- | --- |
| FineCops val | −0.331522 [−0.436092, −0.234780] | +0.916219 [+0.755069, +1.077840] | −1.247741 [−1.366442, −1.137506] |
| gRef source-disjoint | −0.153870 [−0.223410, −0.083454] | +1.043325 [+0.917001, +1.169313] | −1.197195 [−1.305783, −1.088155] |
| gRef Full | −0.157004 [−0.221966, −0.090638] | +1.025586 [+0.906844, +1.140500] | −1.182590 [−1.281130, −1.080634] |

三个seed在所有评价面均给出Y风险降低、E风险升高。两个gRef面重合，不能算两份独立benchmark复现。coverage干预只在MM-GDINO上完成，不能宣称两个架构都验证了覆盖机制。

FineCops四格AUGRC：L1 E=25.881665，L1 Y=26.019433；Full E=26.797885，Full Y=25.687911。Y−E由 +0.137767 [+0.053387,+0.222694] 变为 −1.109974 [−1.278545,−0.950507]。

**1.248不是Y的改善量。** Y的绝对改善是0.332，交互同时包含E的0.916恶化。主文摘要、Introduction、Table2和结论均保留这个区别。

## 排序解释与适用边界

FineCops的Full−L1变化：

| Target | Δ C/W AUROC | Δ C/N AUROC | Δ W/N AUROC | Δ existence AUROC |
| --- | ---: | ---: | ---: | ---: |
| Y | −11.411195 | +4.411990 | +18.281045 | +7.551876 |
| E | −33.392242 | +3.152647 | +27.358527 | +8.632739 |

Y不是恢复了C/W正确性排序，而是C/N收益超过C/W代价。其C/N变化中，cross-level贡献 +5.181690、within-level贡献 −0.769700；L1条件C/N本身降低1.019370点。不能把完整风险改善称为各难度均获益。

Full下Y−E的FineCops C/N仍为 −2.327307 [−2.947569,−1.730074]。固定条件人群下的Y/E AUGRC内部交点由 π=0.434245 [0.401425,0.467227] 移到0.768580 [0.720554,0.818652]。两者全部5,000 draws均有内部交点；图2显示均值曲线和交点区间，而不是完整曲线的置信带。

gRef source-disjoint中Y的C/N改善1.483053点、C/W降低0.656597点，风险改善方向延续但正确性代价较小。Full Y−E的C/N为 +0.193276 [−0.372526,+0.776571]，不能宣称两种pairwise能力全面占优。其均值曲线没有内部交点，但1,275个bootstrap draw仍有内部交点，不给无条件root区间或普遍优势保证。

风险恒等式分解是pairwise贡献，不等同于单独wrong-box/no-target风险积分分量：FineCops DY中C/W贡献 +0.521368、C/N贡献 −0.852889，总和−0.331522；DE对应 +1.525663、−0.609444，总和+0.916219。它们是现有点估计的代数解释，没有额外计算分量CI。

## 论文结构与贡献配合

- Abstract与Introduction：前置受控coverage结果和绝对效应，不再以“下一步检验覆盖”收尾。
- Table1：两个localizer的target/readout对照，回答输出query读出是否足够。
- Table2：coverage×target四格风险、DY/DE/I及冻结迁移。
- Table3：C/W、C/N、W/N与existence/risk的变化，解释为什么排序收益不能互相替代。
- Figure1：从初始现象到读出敏感性，再到覆盖干预；(b)继续显示三个seed和均值，不让seed17驱动的大均值成为稳定机制证据。
- Figure2：coverage改变风险反转的适用人群，而不是消除所有残余损失。
- 组合：只针对原paired-L1 heads的固定组合，作为实践结尾；没有将它们伪装成Full-head融合或新方法贡献。
- 工具、更新/参数收据、完整seed表与大量诊断留在补充材料和仓库，不挤占主文发现的表达空间。

## 文件和复现

- [主文源文件](../paper/empirical_study_v8.tex) / [PDF](../paper/empirical_study_v8.pdf)
- [补充材料源文件](../paper/empirical_supplement_v8.tex) / [PDF](../paper/empirical_supplement_v8.pdf)
- [实验完成收据](../paper/data/coverage_v1/completion.json)
- [FineCops](../paper/data/coverage_v1/analysis/finecops_val.json)、[gRef Full](../paper/data/coverage_v1/analysis/gref_full.json)、[gRef source-disjoint](../paper/data/coverage_v1/analysis/gref_source_disjoint.json)
- [图表生成器](../paper/scripts/build_coverage_v8_assets.py) / [图表收据](../paper/generated/coverage_v8_r2/receipt.json)

```bash
python3 -m pytest -q tests/test_coverage_v8_assets.py
make -C paper empirical-v8-audit
make -C paper current TECTONIC=/path/to/tectonic
```

构建写入 `.build/empirical_v8`；不覆盖原v7.1及更早PDF、checkpoint、records或统计。`generated/coverage_v8`是未发布的首轮排版草稿，最终版本仅使用`coverage_v8_r2`。
