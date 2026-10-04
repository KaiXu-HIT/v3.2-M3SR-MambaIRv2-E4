# E2：Spatial RGB Uncertainty（U1 / U2 / U3）

本仓库从 E1/E0 继续，只替换 UDR 门控中的 RGB uncertainty。Depth encoder、全局 alpha、RGB MambaIRv2 主干计算、重建头、数据预处理和 L1 重建损失均沿用 E0。U1 使用 `1-max(p_route)`；U2 使用最终 RGB feature 的 3×3 局部方差并按每图 P95 缩放；主方案 U3 使用 `Conv3x3(C,C//4) → GELU → DWConv3x3 → GELU → Conv1x1 → Sigmoid`。原有 E0 架构默认仍计算 entropy。

U3 的额外监督使用**固定** RGB-only `net_g_490000.pth` 的预测与 GT 之差：先对 HR RGB 通道取平均绝对误差，再以 4×4 平均池化对齐 LR uncertainty map，按每图 P95 归一并截断到 `[0,1]`。损失为原 L1 重建损失加 `0.01 × L1(U,T_U)`。P95 归一是 E2 文档中未指定 `Norm` 的明确实现约定。

## 服务器环境和权重

以下命令在已安装项目依赖、`mamba_ssm` 和 `selective_scan_cuda` 的 Linux CUDA 环境执行。数据路径完整继承 E0 YAML；若服务器代码目录不同，可在克隆目录运行而不修改数据集路径。两个历史权重的绝对路径应先核对：

```bash
git clone https://github.com/KaiXu-HIT/v3.2-M3SR-MambaIRv2-E2.git
cd v3.2-M3SR-MambaIRv2-E2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
test -f /home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth
test -f /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2/experiments/v3.2_UDR_MambaSR_x4_phaseB/models/net_g_100000.pth
python scripts/udr/check_e2_uncertainty.py
```

## U1/U2 预筛与训练

U1 的 routing 分布可能接近均匀。先用 E0 权重做 U1/U2 无训练筛查，重点看 `E2_statistics.json` 中 `uncertainty_p95 - uncertainty_p05`、`uncertainty_std` 以及与 RGB-only 重建误差的 Spearman 相关。若 U1 几乎常数，不应凭单次 PSNR 将其当作有效 uncertainty。U1/U2 无新增可训练 head，直接从 E0 权重联合微调 Phase B；U3 才有独立的 head-only Phase A。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/e2_uncertainty_audit.py \
  --e2-config options/test/mambairv2/test_E2_U1_x4.yml \
  --checkpoint-kind e0 \
  --e2-checkpoint /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2/experiments/v3.2_UDR_MambaSR_x4_phaseB/models/net_g_100000.pth \
  --seeds 10 11 12 --output results/E2_udrv2_uncertainty/U1_preflight

CUDA_VISIBLE_DEVICES=0 python scripts/udr/e2_uncertainty_audit.py \
  --e2-config options/test/mambairv2/test_E2_U2_x4.yml \
  --checkpoint-kind e0 \
  --e2-checkpoint /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2/experiments/v3.2_UDR_MambaSR_x4_phaseB/models/net_g_100000.pth \
  --seeds 10 11 12 --output results/E2_udrv2_uncertainty/U2_preflight

CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_E2_U1_phaseB_x4.yml
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_E2_U2_phaseB_x4.yml
```

## 主方案 U3 两阶段训练

Phase A 训练 30k，仅 uncertainty head 可训练；RGB backbone 与整个 Depth 分支参数及运行状态均冻结。Phase B 从 A 的完整 E2 权重开始新的优化器，联合训练 100k；head 与 Depth 分支 LR=`1e-4`，RGB backbone LR=`1e-5`，学习率恒定。两个阶段分别启动，不要跨阶段使用 `--auto_resume`。

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_E2_U3_phaseA_x4.yml
test -f experiments/E2_U3_head_only_x4/models/net_g_30000.pth
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_E2_U3_phaseB_x4.yml
```

## 五数据集匹配种子测试

每个命令依次测试 Set5、Set14、B100、Urban100、Manga109，RGB baseline 与实验模型均以相同的 `10 11 12` hard-Gumbel 种子重复推理，输出逐 seed 和 mean±std、配对 ΔPSNR。指标为原有 uint8 Y 通道、×4、crop border 4。下列 E0 对照用于 E2 Strong Go 判定。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_repeated.py \
  --udr-checkpoint /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2/experiments/v3.2_UDR_MambaSR_x4_phaseB/models/net_g_100000.pth \
  --seeds 10 11 12 --output results/E2_udrv2_uncertainty/E0_five_set

CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_repeated.py \
  --udr-config options/test/mambairv2/test_E2_U1_x4.yml \
  --udr-checkpoint experiments/E2_U1_joint_x4/models/net_g_100000.pth \
  --seeds 10 11 12 --output results/E2_udrv2_uncertainty/U1_five_set

CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_repeated.py \
  --udr-config options/test/mambairv2/test_E2_U2_x4.yml \
  --udr-checkpoint experiments/E2_U2_joint_x4/models/net_g_100000.pth \
  --seeds 10 11 12 --output results/E2_udrv2_uncertainty/U2_five_set

CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_repeated.py \
  --udr-config options/test/mambairv2/test_E2_U3_x4.yml \
  --udr-checkpoint experiments/E2_U3_joint_x4/models/net_g_100000.pth \
  --seeds 10 11 12 --output results/E2_udrv2_uncertainty/U3_five_set
```

## 机制审计与判定

以下命令对每个变体使用 DIV2K validation 全量图像、三次 matched seeds，输出 `E2_mechanism.csv`、`E2_statistics.json`、`E2_summary.md` 和 20 张 uncertainty/gate/correction 可视化。比较 E0 的 `E0_statistics.json` 和上面的五数据集报告；若 E0 审计文件尚未生成，可先按 `docs/E0_GUIDE.md` 运行。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/e2_uncertainty_audit.py \
  --e2-config options/test/mambairv2/test_E2_U1_x4.yml \
  --e2-checkpoint experiments/E2_U1_joint_x4/models/net_g_100000.pth \
  --e0-statistics results/E0_udr_mechanism_audit/E0_statistics.json \
  --e0-five-set results/E2_udrv2_uncertainty/E0_five_set/summary.json \
  --e2-five-set results/E2_udrv2_uncertainty/U1_five_set/summary.json \
  --seeds 10 11 12 --output results/E2_udrv2_uncertainty/U1_audit

CUDA_VISIBLE_DEVICES=0 python scripts/udr/e2_uncertainty_audit.py \
  --e2-config options/test/mambairv2/test_E2_U2_x4.yml \
  --e2-checkpoint experiments/E2_U2_joint_x4/models/net_g_100000.pth \
  --e0-statistics results/E0_udr_mechanism_audit/E0_statistics.json \
  --e0-five-set results/E2_udrv2_uncertainty/E0_five_set/summary.json \
  --e2-five-set results/E2_udrv2_uncertainty/U2_five_set/summary.json \
  --seeds 10 11 12 --output results/E2_udrv2_uncertainty/U2_audit

CUDA_VISIBLE_DEVICES=0 python scripts/udr/e2_uncertainty_audit.py \
  --e2-config options/test/mambairv2/test_E2_U3_x4.yml \
  --e2-checkpoint experiments/E2_U3_joint_x4/models/net_g_100000.pth \
  --e0-statistics results/E0_udr_mechanism_audit/E0_statistics.json \
  --e0-five-set results/E2_udrv2_uncertainty/E0_five_set/summary.json \
  --e2-five-set results/E2_udrv2_uncertainty/U3_five_set/summary.json \
  --seeds 10 11 12 --output results/E2_udrv2_uncertainty/U3_audit
```

机制指标优先查看 `std(U)`、`Spearman(U, RGB-error)`、`gate_std` 和 `correction_rms`，再看 PSNR/SSIM。脚本按照方案判定：Spearman>0.2 且相对 E0 明显提高为 Go；这里把“明显提高”明确为至少 `+0.05` Spearman；Spearman>0.3 且五数据集平均 ΔPSNR 高于 E0 为 Strong Go；相关性非正且 PSNR 无改善为 No-Go。缺失的 E0 或五数据集实测结果会保持 pending/inconclusive，不会补造数字。
