# E4：UDR-v2 Full Integration

E4 在原 late residual 位置组合已实测选出的 E2 uncertainty 和 E3 local alpha：

`F' = F_RGB + A_D × U_R × C_D × R_D`。

`U_R` 是 RGB 重建困难度，`C_D` 是沿用的 Depth confidence，`A_D` 是局部干预强度，`R_D` 是沿用的 Depth feature correction。E4 复用 E2 的 U1/U2/U3 实现与 E3 的局部 alpha 实现，不增加网络模块，不改原 RGB routing/Depth 数据处理/五数据集评估口径。源码的新增或组合处均有注释。

## 必需的上游实测结果

目前没有 E2/E3 实测结果和权重，因此仓库**没有预选** `U_best`、`A_best`，也没有虚构性能。先分别按照 [E2 指南](E2_GUIDE.md)、[E3 指南](E3_GUIDE.md)完成三种 U、四档 A 的训练、三种 matched-seed 五数据集测试、DIV2K 机制审计；还需 E0 的 `E0_statistics.json`。E4 选择器要求 E2 的 Spearman `>0.2` 且比 E0 高至少 `0.05`（沿用 E2 指南对“明显提高”的量化），在符合条件者中按五数据集平均 ΔPSNR 选最大；E3 要有非零图内 alpha 标准差与 P95−P05，在符合条件者中按五数据集平均 ΔPSNR 选最大。同分时依次比较 Urban100/Manga109。无有效候选会报错，不会冒充“最佳”。

选择器从 E3 完整权重保留共享 RGB、Depth 与局部 alpha 参数；U1/U2 不含学习参数，U3 只从 E2 完整权重移植训练后的 uncertainty head。共享键及尺寸必须一致，最终 E4 模型还会严格加载合并状态。生成 `experiments/E4_selection/E4_selection.json`、`E4_initial.pth` 和固定名称的 E4 A/B/test YAML。生成 YAML 逐项继承 E3 的原始数据路径和评价设置。

下列命令在 Linux/CUDA 服务器的 E4 项目根目录执行。`E2ROOT` 和 `E3ROOT` 指向已完成实验的各自项目目录，`E0STATS` 指向真实 E0 审计报告。请核对绝对路径；权重若放在其他位置，可向选择器另传 `--e2-checkpoint`、`--e3-checkpoint`。E4 checkout 可以用本仓库或克隆目标 E4 仓库。

```bash
git clone https://github.com/KaiXu-HIT/v3.2-M3SR-MambaIRv2-E4.git
cd v3.2-M3SR-MambaIRv2-E4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
E2ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E2
E3ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E3
E0STATS=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E2/results/E0_udr_mechanism_audit/E0_statistics.json
RGB=/home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth
E0=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2/experiments/v3.2_UDR_MambaSR_x4_phaseB/models/net_g_100000.pth
test -f "$RGB" && test -f "$E0" && test -f "$E0STATS"
python scripts/udr/prepare_e4_integration.py \
  --e2-results "$E2ROOT/results/E2_udrv2_uncertainty" \
  --e3-results "$E3ROOT/results/E3_local_alpha" \
  --e0-statistics "$E0STATS" \
  --rgb-teacher-checkpoint "$RGB"
python scripts/udr/check_e4_udrv2.py
```

## 训练

Phase A 从严格合并权重开始，冻结 RGB，只训练 Depth/局部 alpha 与所选 U3 head（如果为 U3）30k；Phase B 从 A 的完整权重重开优化器，RGB LR=`1e-5`，Depth/alpha/U3 LR=`1e-4`，再训练 100k。损失为原 L1、选中 E3 的 `λ_A mean|A_D|`，以及仅在 U3 情况下继承 E2 的 `0.01 L1(U_R,T_U)`；U3 teacher 是冻结的 RGB-only 模型。A/B 分开启动，不跨阶段 `--auto_resume`。

```bash
test -f experiments/E4_selection/E4_initial.pth
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_E4_phaseA_x4.yml
test -f experiments/E4_selected_phaseA_x4/models/net_g_30000.pth
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_E4_phaseB_x4.yml
test -f experiments/E4_selected_phaseB_x4/models/net_g_100000.pth
```

## 五组消融与测试

以下命令都在 **E4 项目根目录**运行，统一从固定 RGB baseline 加载，使用原 Set5/Set14/B100/Urban100/Manga109、uint8 Y-channel、×4、crop 4、matched seeds `10 11 12`。选择 JSON 读取实际胜出的标签，不猜测变体。`evaluate_repeated.py` 同时计算 RGB 和本组模型，消融汇总器检查每次 RGB 逐 seed 指标完全一致。此处 E0 即 UDR-v1；E2 即 U only；E3 即 A only；E4 即 U+A。

```bash
U=$(python -c "import json; print(json.load(open('experiments/E4_selection/E4_selection.json'))['selected_uncertainty']['variant'])")
A=$(python -c "import json; print(json.load(open('experiments/E4_selection/E4_selection.json'))['selected_alpha']['tag'])")
test -f "$E2ROOT/experiments/E2_${U}_joint_x4/models/net_g_100000.pth"
test -f "$E3ROOT/experiments/E3_${A}_phaseB_x4/models/net_g_100000.pth"
CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_repeated.py \
  --baseline-checkpoint "$RGB" --udr-checkpoint "$E0" \
  --seeds 10 11 12 --output results/E4_udrv2/E0_five_set
CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_repeated.py \
  --baseline-checkpoint "$RGB" \
  --udr-config "options/test/mambairv2/test_E2_${U}_x4.yml" \
  --udr-checkpoint "$E2ROOT/experiments/E2_${U}_joint_x4/models/net_g_100000.pth" \
  --seeds 10 11 12 --output results/E4_udrv2/E2_five_set
CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_repeated.py \
  --baseline-checkpoint "$RGB" \
  --udr-config "options/test/mambairv2/test_E3_${A}_x4.yml" \
  --udr-checkpoint "$E3ROOT/experiments/E3_${A}_phaseB_x4/models/net_g_100000.pth" \
  --seeds 10 11 12 --output results/E4_udrv2/E3_five_set
CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_repeated.py \
  --baseline-checkpoint "$RGB" \
  --udr-config options/test/mambairv2/test_E4_x4.yml \
  --udr-checkpoint experiments/E4_selected_phaseB_x4/models/net_g_100000.pth \
  --seeds 10 11 12 --output results/E4_udrv2/E4_five_set
python scripts/udr/e4_ablation.py
```

`results/E4_udrv2/ablation/E4_ablation.md` 给出 RGB、UDR-v1、U only、A only、U+A 五行表，以及各数据集 ΔPSNR；JSON 含 PSNR/SSIM 的 mean±std 与判据。最低成功线：五集平均 ΔPSNR `>+0.03 dB`，Urban100 `≥0`，Manga109 `>0`；`>+0.05 dB` 较有价值，`+0.08～+0.10 dB` 理想。仅以真实匹配种子测试结果判定。

## 四因子机制审计

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/e4_mechanism_audit.py \
  --rgb-checkpoint "$RGB" \
  --e4-checkpoint experiments/E4_selected_phaseB_x4/models/net_g_100000.pth \
  --seeds 10 11 12 --visualize-count 20 \
  --output results/E4_udrv2/mechanism
```

审计对 DIV2K validation 输出 `U_R`、`C_D`、`A_D`、修正量与 RGB/E4 误差图及逐图统计。请把它与五集消融一同判断，不能由图示推断不存在的性能数字。
