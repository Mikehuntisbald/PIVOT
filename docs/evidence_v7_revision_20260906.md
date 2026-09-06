# v7：从读出交互转向监督覆盖与成功／失败比较

这是对已完成 v6 的行文与解析解释修订。没有新训练、模型 forward、Test 访问或 bootstrap；原始 v6 源码、PDF、统计 JSON、checkpoint 和记录保持不变。

## 本轮主线

相同监督改动为何产生相反风险结果 → 读取输出 query 能否解释 → 三状态排序分解 → 监督覆盖与迁移边界 → 简单组合是否缓解。

“supervision–readout interaction”不再作为标题式贡献。四格风险、`D_emit`、`D_exists` 和 interaction 仍完整报告，作用是检验解释，而不是要求交互必须成立。

## 提升为主要发现的证据

1. **读取实际输出 query 不是充分修复条件。** MM-GDINO FineCops 的 emit−exists AUGRC 劣势在 matched Selected 下仍为 +0.159465 [0.076733, 0.241143] 个点。
2. **仅换推理读出不等于匹配训练。** MDETR FineCops 的 global-emission 头只在推理时改读 Selected，AUGRC 增加 +1.607516 [1.513385, 1.697795]；从相同原始初始化训练并部署 Selected，相对同一 G/G 参考则改变 −0.189079 [−0.256630, −0.122089]。其训练 seed 变异保留在表内；不能解释为唯一的空间因素贡献。
3. **L1 与跨难度结果方向不同。** MM-GDINO global emit−exists：

| 比较 | AUROC 改变量 ×100 | 现有 image-bootstrap 95% CI |
| --- | ---: | --- |
| L1 内 C–W | +7.084626 | [+6.022783, +8.236068] |
| L1 内 C–N | +0.223845 | [+0.024417, +0.431406] |
| 全体 C–N | −3.523202 | [−3.883794, −3.165805] |
| 对全体 C–N 的 within-level 加权贡献 | +0.169020 | [+0.018574, +0.325809] |
| 对全体 C–N 的 cross-level 加权贡献 | −3.692222 | [−4.018634, −3.383512] |

条件 AUROC 改变量与加权贡献不是同一个量。后两行相加得到全体 C–N 改变量。由于所有 negatives 的 positive parent 都是 L1，cross-level 项比较的是 L2/L3 positives 与 L1-parent negatives。负向项集中在这些比较上，不是所有 L1 内的成功排序都变差。

80,451 个训练 pair 只使用 43,979 个唯一 L1 正例父样本；L2/L3 未进入置信度头 loss。这是 difficulty support 的边界，不是说 validation L1 样本就是训练样本。MM-GDINO trunk 的早期 positive-source 训练仍见过完整 positive population；不能混同 trunk 与 head 的覆盖。

## L1 均值曲线的解析推论

条件人群是全部 6,097 个 L1 positives（C=5,506、W=591）与现有全部 9,029 个 no-target requests。令 `a1=5506/6097`，只改变 no-target prior π，保持两类条件分布与 Native 输出固定。

已有三 seed 均值给出 `ΔU_CW=0.07084626339025324`、`ΔU_CN=0.0022384518727516602`。代入已使用的 AUGRC 恒等式：

`ΔR1(π)=−(1−π)a1[(1−π)(1−a1)ΔU_CW+πΔU_CN]`。

两项都为正，因此对 `0≤π<1`，固定均值风险差严格为负；`π=1` 时为零。这个均值曲线没有内部 crossover。

这是**现有 pairwise point estimates 的事后解析推论**，不是新增训练结果、已计算的 L1 risk CI 或 simultaneous confidence band。两个边际 AUROC 区间不能直接代替完整曲线的不确定性。推论也不证明覆盖不足是因果来源；本轮没有改变训练覆盖。

新 [解析收据](../paper/generated/evidence_v7_r2/analytic_l1.json) 将 `risk_ci95=null`、`new_bootstrap_replicates=0` 与固定人群计数显式绑定。测试覆盖同向改善、相反方向产生交点、退化条件与均值代数。

## 全文改动

- Abstract / Introduction：突出相反风险结果、输出 query 不充分、L1 正向比较和跨难度负向项。
- Figure 1：相反风险 → inference-only 与 matched training → 全体与 L1 的 C–N 差异；interaction 不再占据视觉中心。
- 主文加入 MDETR 交叉读出表和 L1/全体/加权贡献表，避免诊断只留在 supplement。
- 将重复的防御性句子集中到方法与边界说明，正文优先讲结果与解释线索。
- 保留 frozen transfer 的方向变化、seed 异质性以及 AURC/AUGRC 的区别。
- Product 以实际收益与代价收束：MM-GDINO gRef 可优于三个参考，FineCops 则不如两个学习头，不包装成新方法贡献。

## 文件与复现

- [v7 主文](../paper/empirical_study_v7.tex) / [PDF](../paper/empirical_study_v7.pdf)
- [v7 补充材料](../paper/empirical_supplement_v7.tex) / [PDF](../paper/empirical_supplement_v7.pdf)
- [v6 完整结果](readout_v6_final_results_20260906.md)仍是原始数值入口。
- [新资产生成器](../paper/scripts/build_evidence_v7_assets.py)只读取 sealed 三套 JSON 和原始 renderer；不加载权重或 records。

```bash
python3 -m pytest -q tests/test_evidence_v7_assets.py
make -C paper empirical-v7-audit
make -C paper empirical-v7 empirical-supplement-v7 TECTONIC=/path/to/tectonic
```

下一项科学问题是 coverage-controlled intervention，而不是把现有覆盖分解称为已识别的原因。本轮只完成论文与解析解释修订，不启动该实验。
