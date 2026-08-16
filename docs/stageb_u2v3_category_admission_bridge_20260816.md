# Stage-B U2-v3 category-admission bridge（2026-08-16）

## 结论

在新的 Stage-A/C100 单 checkpoint 骨架上，恢复 category-admission 投影参数训练是有效的，但当前 D2/Gap3 训练目标尚未超过 legacy U2。U25 是本轨迹的 val3 最优点；U50 在三个 split 上一致回退，因此按预注册的 val-only early-stop 原则停止 U100，且不运行 Ref test 或 strict TN。

U25 相对未训练 C0 的 val3 micro 从 `0.702507` 提升到 `0.709944`（`+0.007437`），同时 RefCOCO 和 RefCOCO+ 均提升；但 RefCOCOg 从 `0.804739` 回退到 `0.803105`，所以 U25 不满足“每个 split 不低于 C0/B58”的正式候选门槛。legacy U2 的 val3 micro 仍为 `0.723460`。

## 冻结与梯度所有权

- 初始化器：`outputs/u2v2_diagnostic_20260816/c0/checkpoint_u2v2_c0.pth`
- 初始化器 SHA256：`2578c62a187948e7e459afba0f2c72d3de6901912abc3ac7770a19b86f177309`
- 冻结：B58 trunk、R100 rank8、C100 confidence12、U0 compatibility shell。
- 可训练：Stage-A category-admission 的 8 个 projection/norm tensor，共 263,680 参数。
- 冻结 tensor：1,157 个；聚合 SHA256 在 U1、U25、U50 均为 `c26b72ca29ffa00497a9ec3f46ba6afb8ec399528ea654e9d6ebda7b3f4097a2`。
- 原始 GDINO text branch 始终输入完整 referring expression；训练时不开 hard gate，criterion 在完整 physical B38 上模拟标准化 Gap3 边界；部署评估固定 hard Gap3。
- confidence 与 patch 之外的分支没有 autograd connection；本实验没有训练 post-gate residual。

## 训练协议

- seed 42；D2 三源 row-locked mix `2:2:1`。
- physical batch 38，gradient accumulation 2，forward microbatch 8。
- LR `5e-5`，weight decay `1e-4`，clip `0.1`，AMP initial scale `8`。
- deterministic 800/max1333、no flip。
- U25/U50 均为 0 AMP skip、0 nonfinite gradient、0 zero-gradient successful step。
- U50 累计峰值 allocated `6,901,618,176` bytes；最低 device free `14,473,756,672` bytes。

Checkpoint 与审计：

- U25：`outputs/u2v3_bridge_20260816/formal_seed42_b38a2_gap3_micro8/milestones/checkpoint_iter_000025.pth`，SHA256 `4f043dedea3257757ffe897a070223048a5e04dea46f75641a92d9c7e2c3f71d`。
- U25 audit：同目录 `audit_iter_000025.json`，status `verified`。
- U50：`outputs/u2v3_bridge_20260816/formal_seed42_b38a2_gap3_micro8/milestones/checkpoint_iter_000050.pth`，SHA256 `40842b7ac09ac557475ac033aac090c76b951d5664c6b92ecec7ebdddcc331fe`。
- U50 audit：同目录 `audit_iter_000050.json`，status `verified`。

## Val-only 结果

所有行使用相同三个 val manifest、seed 42、AMP、B16 和固定 Gap3；micro 按 26,488 个 expression 汇总。

| 路由 / milestone | RefCOCO val | RefCOCO+ val | RefCOCOg val | val3 micro |
|---|---:|---:|---:|---:|
| B58 | 0.645468 | 0.694832 | 0.805147 | 0.695032 |
| C0 + Gap3 | 0.650545 | 0.708310 | 0.804739 | 0.702507 |
| U2-v3 U25 + Gap3 | **0.655344** | **0.722532** | **0.803105** | **0.709944** |
| U2-v3 U50 + Gap3 | 0.652575 | 0.722067 | 0.800245 | 0.708094 |
| legacy U2 + Gap3 | 0.666236 | 0.743075 | 0.806985 | 0.723460 |

U25 records/summary：
`outputs/u2v3_bridge_20260816/formal_seed42_b38a2_gap3_micro8/evaluations/u25_val3_gap3_b16_v2/summary.json`

U50 records/summary：
`outputs/u2v3_bridge_20260816/formal_seed42_b38a2_gap3_micro8/evaluations/u50_val3_gap3_b16/summary.json`

## 选择决定

1. U25 优于 U50：U50 三个 split 分别相对 U25 回退 `-0.002769/-0.000465/-0.002860`，micro 回退 `-0.001850`。
2. U25 不是正式候选：RefCOCOg 低于 C0 `-0.001634`、低于 B58 `-0.002042`，且 micro 低于 legacy U2 `-0.013516`。
3. 停止同配置 U100，不查看 test/strict。C100 confidence 的正式逐记录 parity 也留到出现合格 Ref milestone 后再验证，避免无必要消耗 sealed evaluation。
4. 这条结果支持 category-admission 是有效增益源，但说明当前 D2 loss 对 RefCOCO/RefCOCO+ 的优化速度明显快于 RefCOCOg。下一轮应优先做跨 split preserve/采样消融，而不是增加相同目标的 update 数。

## 实现提交

- `a0de43b`：U2-v3 category-admission bridge、配置、契约、审计和测试。
- `f06867a`：micro-forward 后在完整 B38 上一次性计算 criterion，保持 full-batch class balance。
- `7c78cd3`：evaluation loader 将显式 U2-v3 checkpoint 分派到 8/1157 ownership contract，同时保持旧 U2-v2 schema 行为。

权重与 evaluation records 位于 `outputs/`，不提交到 Git；本文仅绑定路径、SHA 和结果。
