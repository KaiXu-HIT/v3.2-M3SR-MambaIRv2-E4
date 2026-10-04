# E3：Local Adaptive Correction Strength

E3 **只改变 correction strength**：从 E0 的全局标量 `alpha` 改为 `1×H×W` 的局部 `A_D=alpha_max×sigmoid(f(F_RGB,F_D,∇D,C_D))`。E0 的 routing-entropy uncertainty、Depth confidence、Depth encoder、residual projection、RGB backbone、SR reconstruction head、数据路径和主 L1 重建损失均保留。E2 的 uncertainty 方案不在 E3 中叠加；组合实验属于 E4。

局部预测器先用 1×1 卷积压缩 RGB/Depth 特征，再拼接 Depth Sobel gradient 和 confidence，经 3×3 卷积、GELU、深度可分离 3×3 卷积、GELU、1×1 卷积得到单通道图。末层偏置使初始均值约 `0.0075`，落在方案的 `0.005–0.01` 范围。E0 的 `alpha_raw` 仅作为严格权重迁移的兼容张量保留并冻结；E3 correction 不再使用它。源码中对此和每处实验改动均有注释。

## 训练设计

四个独立正则组：`a0=0`、`a1e5=1e-5`、`a1e4=1e-4`（主组）、`a5e4=5e-4`。损失是原 L1 加 `λ_A×mean(|A_D|)`。每组 Phase A 从**相同的 E0 Phase B 权重**启动：冻结 RGB，仅训练 local-alpha head 与原 Depth 分支 30k；Phase B 从本组 A 的完整权重启动新优化器，联合训练 100k，Depth/新模块 LR=`1e-4`，RGB LR=`1e-5`。各阶段不跨阶段 `--auto_resume`。同组 A/B 的数据、主损失和架构只差训练参数范围。

以下命令假定服务器已安装项目依赖、`mamba_ssm`、`selective_scan_cuda` 且 CUDA 可用。原始数据路径仍在 YAML 中。先核对 RGB 和 E0 权重绝对路径：

```bash
git clone https://github.com/KaiXu-HIT/v3.2-M3SR-MambaIRv2-E3.git
cd v3.2-M3SR-MambaIRv2-E3
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RGB=/home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth
E0=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2/experiments/v3.2_UDR_MambaSR_x4_phaseB/models/net_g_100000.pth
test -f "$RGB"
test -f "$E0"
python scripts/udr/check_e3_local_alpha.py
```

运行四个独立组的两阶段训练。主组若只需验证推荐值，可单独执行 `a1e4` 的两条命令；完整 E3 正则搜索运行全部四组。

```bash
for TAG in a0 a1e5 a1e4 a5e4; do
  CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt "options/train/mambairv2/train_E3_${TAG}_phaseA_x4.yml"
  test -f "experiments/E3_${TAG}_phaseA_x4/models/net_g_30000.pth"
  CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt "options/train/mambairv2/train_E3_${TAG}_phaseB_x4.yml"
done
```

## 匹配种子测试

固定 RGB baseline 与 E0 对照使用相同的五数据集、uint8 Y-channel、×4、crop border 4。ASSM 在 eval 时仍有 hard-Gumbel 随机性，所有比较均使用 matched seeds `10 11 12`，记录每数据集 mean±std 和配对 ΔPSNR；不要把单次 0.01 dB 当成可靠改进。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_repeated.py \
  --baseline-checkpoint "$RGB" --udr-checkpoint "$E0" \
  --seeds 10 11 12 --output results/E3_local_alpha/E0_five_set

for TAG in a0 a1e5 a1e4 a5e4; do
  CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_repeated.py \
    --baseline-checkpoint "$RGB" \
    --udr-config "options/test/mambairv2/test_E3_${TAG}_x4.yml" \
    --udr-checkpoint "experiments/E3_${TAG}_phaseB_x4/models/net_g_100000.pth" \
    --seeds 10 11 12 \
    --output "results/E3_local_alpha/${TAG}_five_set"
done
```

## 机制审计

每组对 DIV2K validation 全量图像做三次匹配种子推理，输出 `E3_local_alpha.csv`、`E3_statistics.json`、`E3_summary.md` 和 20 张包含 alpha/gate/correction 的面板。重点查看 `local_alpha_mean/std/p05/p50/p95/max`、`correction_rms`、`correction_active_ratio`（`A_D>0.01`），以及五数据集平均 ΔPSNR 和 Urban100 相对 E0 的变化。目标是局部强度有空间分化，并在有益区域放大、有害区域减小，而非要求所有位置更强。全局 E0 alpha 的图内标准差按定义为零；E3 报告 P95−P05 和 std，但不能只靠这两个数声称改进，需结合可视化和 matched-seed 性能。

```bash
for TAG in a0 a1e5 a1e4 a5e4; do
  CUDA_VISIBLE_DEVICES=0 python scripts/udr/e3_local_alpha_audit.py \
    --rgb-checkpoint "$RGB" \
    --e3-config "options/test/mambairv2/test_E3_${TAG}_x4.yml" \
    --e3-checkpoint "experiments/E3_${TAG}_phaseB_x4/models/net_g_100000.pth" \
    --e0-five-set results/E3_local_alpha/E0_five_set/summary.json \
    --e3-five-set "results/E3_local_alpha/${TAG}_five_set/summary.json" \
    --seeds 10 11 12 \
    --output "results/E3_local_alpha/${TAG}_audit"
done
```

仓库不预填不存在的训练或性能数字。上述训练、五数据集测试、机制审计必须在具有权重和数据的服务器上实际运行。
