# E0：UDR v3.2 机制审计

本实验**不训练新模型**，只读取固定的 RGB baseline `net_g_490000.pth` 和已经完成训练的 UDR Phase B `net_g_100000.pth`，在 DIV2K validation 上审计实际推理行为。原训练、验证、测试数据路径均由已有 YAML 读取。

## 已完成的源码核查

- ASSM 的 `pred_route` 是 `route` 网络输出的 **LogSoftmax**，并非未经归一化的原始 logits。UDR 在 hard Gumbel 采样和语义排序前调用 `softmax(pred_route)`，得到 128 类 routing probability，按 `−Σp log(p)/log(128)` 生成每个 LR token 的模糊度。
- UDR 对 ASSB4、5、6 中各 ASSM 的模糊度先做层内均值，再做三个 stage 等权均值。该值是 routing ambiguity proxy，源码本身没有将其校准为重建误差概率。
- Depth confidence 来自 RGB 亮度梯度与归一化 Depth 梯度的四通道 GRE；`gate = ambiguity × confidence`。Depth residual 来自 Depth encoder 与 RGB context 融合后的投影。
- 最终 correction 是 `alpha × gate × depth_residual`，位于 ASSB6 之后、`conv_after_body` 之前。`alpha = 0.1 × tanh(alpha_raw)`，初始值为 0.01；训练后的数值只能从对应 checkpoint 实测。
- eval 仍执行 `F.gumbel_softmax(..., hard=True)`，因此 E0 使用匹配种子并报告每图每 seed 的结果。以上核查确认了计算路径；`ambiguity≈0.99994` 与 `alpha≈0.003` 的实际原因及空间相关性，必须由训练完成的权重和 DIV2K validation 输出判断。

## 服务器运行命令

在已有 MambaIRv2 CUDA 环境中执行。仓库更名后首次克隆：

```bash
git clone --branch main https://github.com/KaiXu-HIT/v3.2-M3SR-MambaIRv2-E0.git /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E0
cd /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

如已经克隆，请先进入现有目录，运行 `git pull --ff-only`。仓库更名不要求服务器目录同时改名。先运行代码自测与服务器数据/权重检查：

```bash
python scripts/udr/e0_mechanism_audit.py --self-test
CUDA_VISIBLE_DEVICES=0 python scripts/udr/check_udr.py --check-data
```

完整 E0 审计（全部 DIV2K validation 图像，matched seeds 10/11/12，首 20 张图的 seed=10 可视化）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/e0_mechanism_audit.py \
  --udr-checkpoint /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2/experiments/v3.2_UDR_MambaSR_x4_phaseB/models/net_g_100000.pth \
  --seeds 10 11 12 --visualize-count 20 \
  --output results/E0_udr_mechanism_audit
```

上面的 UDR 路径指向原 v3.2 训练目录；如果权重位于其他位置，请只调整 `--udr-checkpoint`。RGB baseline 默认沿用现有配置中的固定路径：

```text
/home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth
```

可通过 `--rgb-checkpoint /实际/路径/net_g_490000.pth` 覆盖读取位置。切勿用新训练的 RGB 权重替代固定基线。正式 E0 不需要运行 `basicsr/train.py`。

如需先快速检查脚本与服务器环境（不产生正式 E0 结论）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/e0_mechanism_audit.py \
  --udr-checkpoint /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2/experiments/v3.2_UDR_MambaSR_x4_phaseB/models/net_g_100000.pth \
  --seeds 10 --max-images 2 --visualize-count 2 \
  --output results/E0_smoke
```

## 输出与统计口径

正式命令生成以下四项：

```text
results/E0_udr_mechanism_audit/E0_mechanism_audit.csv
results/E0_udr_mechanism_audit/E0_statistics.json
results/E0_udr_mechanism_audit/E0_visualization/
results/E0_udr_mechanism_audit/E0_summary.md
```

CSV 为逐图、逐 seed 记录，包含路由熵、最大概率、概率 margin、模糊度各分位数、Depth confidence、gate、alpha、残差的 RMS/绝对值、RGB baseline 重建误差、Pearson/Spearman 及两模型 Y 通道 PSNR/SSIM。JSON 包含各字段的均值/中位数/标准差、有效相关系数样本数和首次分块中的张量形状。常数图的相关系数未定义，写为 `null`/空值，不伪造为 0。可视化每张包含 LR RGB、GT、RGB-only SR、UDR SR、RGB error、Depth、Depth gradient、ambiguity、confidence、gate、correction magnitude。

E0 在 **LR 网格**比较空间图：将 HR 的 RGB-only 绝对误差按通道平均，再对每个 ×4 区块取平均。这个误差图用于机制相关性；PSNR/SSIM 独立使用原有 uint8、Y 通道、裁边 4 的评测口径。图像分块、重叠与原模型测试相同；每张图在 RGB 与 UDR 推理前分别重置相同的 Gumbel seed。

脚本通过临时 forward hook 读取实际 `pred_route`（`LogSoftmax` 输出）、`softmax(pred_route)` 概率、UDR 的 ambiguity/confidence/gate、depth residual 与 correction。每个分块还会核对读出的 entropy 是否等于真实融合所用的 ambiguity，核对 `correction = alpha × gate × residual`。它不会修改权重、模型前向逻辑或训练配置。

`E0_summary.md` 会按方案给出一个工作判断：Spearman > 0.2 且 U 的 P95−P05 ≥ 0.05 时保留 routing proxy；0.05–0.2 提示校准；相关性绝对值 ≤ 0.05 且空间 std < 0.01 时放弃把 routing entropy 解释为 RGB 重建不确定性。动态范围阈值是 E0 明确采用的操作口径，需结合 20 张图的空间模式检查。**alpha 下降是否由负迁移造成不能仅凭一次 checkpoint 证明**；此实验仅报告该假设的描述性证据。

## 本地验证边界

本地已用可微分参考 scan 验证 E0 钩子及所有地图关系，并用合成图验证奇数尺寸分块、常数相关性和 CSV 字段计算。当前 Windows 环境缺少 `mamba_ssm`，且没有服务器 DIV2K 验证集和训练完成的 Phase B 权重，因此四项实测输出需在服务器执行正式命令后生成。
