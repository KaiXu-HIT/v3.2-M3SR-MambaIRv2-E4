# UDR-MambaSR v3.2：实现与完整运行命令

本地源码工作空间：`D:\Code\Python\M3SR-MambaIRv2`。
保存仓库：<https://github.com/KaiXu-HIT/v3.2-M3SR-MambaIRv2-E0>，分支 `main`。
方案原文：[UDR_MambaSR_SPEC.md](UDR_MambaSR_SPEC.md)。

## 已确认的实验配置

| 项目 | Phase A | Phase B |
|---|---|---|
| 迭代数 | 100,000 | 100,000（独立计数） |
| 初始权重 | 已完成训练的 RGB baseline | Phase A 的完整 100k 权重 |
| RGB 主干与重建头 | 冻结参数并保持 eval 行为 | 解冻，LR = 1e-5 |
| UDR 分支 | LR = 1e-4 | LR = 1e-4 |
| 学习率变化 | 恒定，无 warmup | 恒定，无 warmup |
| 损失 | mean L1，权重 1 | mean L1，权重 1 |
| 验证和保存间隔 | 5,000 | 5,000 |

已确认的 RGB checkpoint（读取 `params`）：

```text
/home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth
```

训练 DIV2K、验证 DIV2K、测试 Set5/Set14/B100/Urban100/Manga109 的全部 `datasets` 设置直接继承原 GTSS 配置，包括所有路径、文件模板、归一化、batch size 和数据增强设置。原有配置文件未被改写。

## 方案到源码的对应关系

| 方案 | 实现 |
|---|---|
| RGB 主干不受 Depth 干预 | `UDRMambaIRv2` 沿用 `MambaIRv2` 的参数名与主干计算；传入各 ASSB 的参数不含 Depth，未创建 GTSS controller |
| RGB routing ambiguity | `mambairv2_arch.py` 的 ASSM 在 Gumbel 采样和排序前，只读原 route 输出，计算 FP32 归一化熵；不增加随机数消耗 |
| ASSB4–6 聚合 | 每个 stage 内对六个 ASSM 的熵取平均，再对三个 stage 等权平均；保留原空间顺序 |
| P2/P98 深度归一化 | 复用 `RGBDepthPairedImageDataset`，完整 LR 图上完成归一化，再裁剪；分块推理不重复归一化 |
| 深度可靠性 | 共享尺度的固定 Sobel/8 幅值；构造 `[E_R,E_D,abs(E_R-E_D),E_R*E_D]`；GRE 为 4→16→8→1 + sigmoid |
| 轻量几何编码 | `[D_N,E_D]` 经 Conv 2→32、GELU、DWConv、GELU、PWConv |
| 残差专家 | 对 RGB context 做逐像素通道 LayerNorm，与 32 通道 Depth 特征拼接，DWConv + 1×1 projection 回到 C 通道 |
| 唯一融合位置 | ASSB6 和原 final norm 后、`conv_after_body` 前；原 final norm 保留，修正结果进入原 global residual 和 SR head |
| 双条件门 | `G=U_RGB*C_D`；任一为零时残差严格关闭 |
| 有界干预 | `alpha=0.1*tanh(alpha_raw)`，初始 alpha=0.01，projection 权重 std=0.001、bias=0；两者都非零以保证首步梯度 |
| 两阶段训练 | 冻结在 DDP 包装前执行；A 仅优化 `udr.*`，B 为 Depth/RGB 建立独立 LR 参数组 |
| 权重兼容 | A 严格检查全部 RGB key 和 shape，只允许 UDR 新权重未加载；B、测试及断点恢复严格加载完整 UDR |
| 推理比较 | 与原 baseline 相同的分块、重叠、裁边 4、Y 通道 PSNR/SSIM；种子 10/11/12，每个模型构建后和每个数据集前重新设种子 |

`U_RGB` 仅为 routing/representation ambiguity proxy，不是经过校准的 SR reconstruction uncertainty。GRE 由最终 L1 学习，没有额外可靠性标签或损失。alpha 限制系数幅度，并不从数学上保证最终图像误差有界。

新模型共 23,092,885 个参数，其中 `udr.*` 为 42,172 个。日志包含 `ambiguity_mean`、`confidence_mean`、`gate_mean`、`alpha` 和 `correction_rms`。保留历史 GRS/GTSS 源码与配置，仅用于独立历史比较；UDR 不走其 depth→scan 路径。

## 服务器准备

以下为 Linux CUDA 服务器 Bash 命令，沿用已经成功训练 RGB baseline 的 Python/Mamba 环境。当前 `/home/BRAIN/...` 数据路径保持原样，Windows 本地目录用于源码编辑；不要在 Windows 上直接使用这些服务器数据路径启动训练。

首次拉取：

```bash
git clone --branch main https://github.com/KaiXu-HIT/v3.2-M3SR-MambaIRv2-E0.git /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2
cd /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

已拉取过则在该项目目录执行 `git pull --ff-only origin main`。激活原 RGB baseline 使用的环境后，验证真实 CUDA kernel、模型和服务器数据：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/check_udr.py --check-data
```

该命令同时检查已确认的 RGB baseline 权重是否匹配完整生产模型、所有数据配对文件是否存在，以及每库首/中/末样本是否可解码。它不执行正式训练。纯本地参考验证可用 `python scripts/udr/check_udr.py --cpu-reference`，该模式不能代替真实 CUDA 和服务器数据检查。

## 单卡完整训练

按顺序执行；Phase A 成功后才执行 Phase B：

```bash
cd /home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_UDR_MambaSR_x4_phaseA.yml
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_UDR_MambaSR_x4_phaseB.yml
```

两个阶段的模型输出分别为：

```text
experiments/v3.2_UDR_MambaSR_x4_phaseA/models/net_g_100000.pth
experiments/v3.2_UDR_MambaSR_x4_phaseB/models/net_g_100000.pth
```

B 配置已指向 A 的 100k 权重。B 开始时建立新优化器和新迭代计数，不读取 A 的 `.state`。两阶段共 200k 次优化，不是从随机初始化重新训练 500k。

## 断点恢复（同一阶段内）

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_UDR_MambaSR_x4_phaseA.yml --auto_resume
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_UDR_MambaSR_x4_phaseB.yml --auto_resume
```

只运行当前中断阶段对应的命令。BasicSR 将从该阶段实验目录找最新 `.state`，恢复完整 UDR 权重、优化器和 scheduler；A 恢复时也不再加载纯 RGB checkpoint。不要用 A 的 `.state` 恢复 B。

## 多卡训练（可选，两卡示例）

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 basicsr/train.py -opt options/train/mambairv2/train_UDR_MambaSR_x4_phaseA.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 basicsr/train.py -opt options/train/mambairv2/train_UDR_MambaSR_x4_phaseB.yml --launcher pytorch
```

每卡 batch size 仍为 2，故两卡全局 batch size 为 4；严格复现实验优先使用默认单卡设置。使用 DDP，不使用 DataParallel。

## 五库单次测试

RGB baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/mambairv2/test_UDR_RGB_reference_x4.yml
```

最终 Phase B 模型：

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/mambairv2/test_UDR_MambaSR_x4.yml
```

可选查看 Phase A 效果：

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/mambairv2/test_UDR_MambaSR_x4.yml --force_yml name=test_v3.2_UDR_phaseA_x4 path:pretrain_network_g=experiments/v3.2_UDR_MambaSR_x4_phaseA/models/net_g_100000.pth
```

## 正式结果：三次匹配种子重复推理

默认比较 RGB 与最终 UDR，不需要重新训练：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_repeated.py --seeds 10 11 12 --output results/udr_comparison
```

同时比较 v1.0 RGB、v3.0 GRS、v3.1 GTSS 与 v3.2 UDR 时，将下面两个历史权重路径替换为实际文件：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_repeated.py \
  --grs-checkpoint /absolute/path/to/v3.0/net_g_500000.pth \
  --gtss-checkpoint /absolute/path/to/v3.1/net_g_500000.pth \
  --seeds 10 11 12 --output results/udr_four_model_comparison
```

历史权重路径没有在本次会话中确认，因此不伪造默认地址；仅有 RGB 和 UDR 权重即可运行默认正式比较。

输出 `summary.json` 和 `summary.md`，包含逐库 PSNR/SSIM mean ± sample std（ddof=1）、对 RGB 的 PSNR 差、五库等权平均差以及三次配对差。

最低成功标准：五库平均 ΔPSNR > 0；进一步目标 ≥ +0.05 dB，并观察 Urban100/Manga109 的 +0.05–+0.10 dB 稳定提升。脚本将“稳定正增益”明确操作化为三个配对种子的 ΔPSNR 均 > 0，这不是统计显著性检验。若平均差 ≤ 0 且两个复杂结构库均未达到该稳定条件，记录停止规则触发，按方案重新评估 Depth 数据价值，不自动开始新架构搜索。

## 验证边界

已完成的本地验证见 [UDR_VERIFICATION.md](UDR_VERIFICATION.md)。本次交付是源码、配置、测试工具与运行命令；尚未执行服务器上的 200k 正式训练与五库性能评估，因此不声称已经超过 RGB baseline。
