# E1：Manga109 与 Urban100 逐图诊断

E1 是对已训练 E0 模型的诊断实验，**不重新训练或改动模型权重**。直接读取固定 RGB baseline `net_g_490000.pth` 和 UDR Phase B `net_g_100000.pth`，使用已有 YAML 的 Manga109/Urban100 数据路径、RGB/Depth 配对与测试指标。项目源码中所有数据集路径保持原样。

## 运行环境与命令

在具备 `mamba_ssm`、`selective_scan_cuda`、CUDA、上述两个权重及原始数据集的 Linux 服务器上，从项目根目录执行。请将 UDR 权重绝对路径核对为服务器上真实存在的位置；下列路径沿用 E0 说明中的原路径。

```bash
cd /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2v3.2-E1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# E1 训练命令：无。E1 仅分析固定 E0 权重。
CUDA_VISIBLE_DEVICES=0 python scripts/udr/e1_dataset_diagnosis.py \
  --rgb-checkpoint /home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth \
  --udr-checkpoint /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2/experiments/v3.2_UDR_MambaSR_x4_phaseB/models/net_g_100000.pth \
  --seeds 10 11 12 \
  --rgb-edge-threshold 0.05 --depth-edge-threshold 0.05 \
  --output results/E1_dataset_diagnosis
```

上述是完整的 E1 测试命令：一次运行全部 Manga109/Urban100 测试图、三个匹配随机种子、逐图统计、相关性和 120 张可视化面板。若项目根目录不同，仅调整 `cd` 和权重绝对路径；不要改动 YAML 中保留的数据集路径。运行前可以检查文件：

```bash
test -f /home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth
test -f /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2/experiments/v3.2_UDR_MambaSR_x4_phaseB/models/net_g_100000.pth
python scripts/udr/e1_dataset_diagnosis.py --self-test
```

## 统计定义与输出

- 每次 RGB baseline 与 UDR 推理前重置相同随机种子，保持 hard Gumbel routing 的可比性；分块/重叠/拼接沿用 E0 的 `infer_partitioned`。以每图三个种子的平均 ΔPSNR 排序。逐图 CSV 同时保留三次推理的标准差，逐种子原始表另存。
- PSNR/SSIM 使用原测试 YAML 的 uint8 Y 通道、HR `crop_border=4`、×4 设置。`delta_psnr = udr_psnr - baseline_psnr`，SSIM 同理。
- 在完整 LR RGB/Depth 上使用与生产 UDR 相同的 `/8` Sobel 和 replicate padding。阈值默认分别为 `0.05`；边缘密度为超过阈值的像素比例。`depth_rgb_edge_alignment = #(RGB 边缘 ∩ Depth 边缘) / #(RGB 边缘)`。无 RGB 边缘时记为 null。另输出 Depth 边缘未匹配比例，辅助辨别伪深度边缘。
- `repetition_score` 为 LR RGB 边缘图横/纵方向 4–16 像素位移的最大自相关，仅是重复纹理代理指标。`mean_uncertainty` 是 E0 的 routing ambiguity；`mean_confidence`、`mean_gate`、`mean_correction` 直接来自不修改模型的 forward hooks，后者为 feature residual 的通道平均绝对值。
- `E1_per_image.csv` 含实验要求的全部逐图字段；`E1_per_seed.csv` 保存各随机种子；`E1_correlation.json` 分别报告两个数据集上 ΔPSNR 与边缘对齐、置信度、修正强度、Depth 边缘密度等指标的 Pearson/Spearman 相关及有效样本数；`E1_summary.md` 概述统计和判读规则。
- 每套数据按平均 ΔPSNR 取改善最多 20、退化最多 20、中位区间 20，生成 `E1_top20_manga/`、`E1_bottom20_manga/`、`E1_median20_manga/`、`E1_top20_urban/`、`E1_bottom20_urban/`、`E1_median20_urban/`。每张图包含 RGB/Depth、Sobel、GT、两种 SR、两种误差、uncertainty、confidence、gate、correction。面板显示第一个种子作为代表，标题标明排序用的跨种子平均 ΔPSNR。

不能仅凭相关性认定退化原因：应检查 Urban100 的底部样本是否同时具有重复结构、高 Depth 边缘密度、低边缘对齐，以及置信度/门控是否仍然放行；对比 Manga109 顶部样本的真实轮廓。真实实验结果必须在有权重和数据的服务器上运行后才能填写，仓库不预填数字。
