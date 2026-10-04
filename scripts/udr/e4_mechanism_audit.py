"""Audit E4's four-factor U_R, C_D, A_D, R_D late correction maps."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.udr.e0_mechanism_audit import (aggregate, correlation,
                                            infer_partitioned, load_config,
                                            load_weights)
from scripts.udr.e3_local_alpha_audit import map_statistics, seed_all

FIELDS = ('uncertainty_mean', 'uncertainty_std',
          'uncertainty_rgb_error_pearson', 'uncertainty_rgb_error_spearman',
          'confidence_mean', 'confidence_std', 'gate_std',
          'local_alpha_mean', 'local_alpha_std', 'local_alpha_p05',
          'local_alpha_p50', 'local_alpha_p95', 'local_alpha_max',
          'correction_rms', 'correction_active_ratio',
          'delta_psnr', 'delta_ssim')


def visualize(path, sample, rgb_sr, e4_sr, maps, delta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gt = sample['gt'].unsqueeze(0)
    def rgb(tensor):
        return tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
    panels = [
        ('LR RGB', rgb(sample['lq'].unsqueeze(0)), None),
        ('Depth', sample['depth'].squeeze().numpy(), 'viridis'),
        ('GT', rgb(gt), None),
        ('RGB-only SR', rgb(rgb_sr), None),
        ('E4 SR', rgb(e4_sr), None),
        ('RGB |SR-GT|', (rgb_sr-gt).abs().mean(1).squeeze().numpy(), 'magma'),
        ('E4 |SR-GT|', (e4_sr-gt).abs().mean(1).squeeze().numpy(), 'magma'),
        ('U_R', maps['ambiguity'], 'viridis'),
        ('C_D', maps['confidence'], 'viridis'),
        ('U_R × C_D', maps['gate'], 'viridis'),
        ('A_D', maps['local_alpha'], 'viridis'),
        ('|A_D U_R C_D R_D|', maps['correction_abs'], 'magma')]
    fig, axes = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    for ax, (label, value, cmap) in zip(axes.flat, panels):
        ax.imshow(value, cmap=cmap, vmin=0 if cmap else None,
                  vmax=(.1 if label == 'A_D' else 1 if cmap == 'viridis' else
                        float(np.percentile(value, 99)) + 1e-12 if cmap else None))
        ax.set_title(label, fontsize=9)
        ax.axis('off')
    fig.suptitle(f'E4 ΔPSNR {delta:+.4f} dB (representative seed)')
    fig.savefig(path, dpi=125)
    plt.close(fig)


def run(args):
    from basicsr.archs.e4_udrv2_arch import E4UDRMambaIRv2
    from basicsr.archs.mambairv2_arch import MambaIRv2
    from basicsr.data import build_dataset
    from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
    from basicsr.utils.img_util import tensor2img

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Production E4 audit needs CUDA Mamba extensions.')
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use at least three distinct matched seeds.')
    rgb_cfg, e4_cfg = load_config(args.rgb_config), load_config(args.e4_config)
    data_cfg = load_config(args.data_config)
    if rgb_cfg['network_g']['type'] != 'MambaIRv2' or e4_cfg['network_g']['type'] != 'E4UDRMambaIRv2':
        raise ValueError('Expected fixed RGB baseline and selected E4 model.')
    for cfg in (rgb_cfg, e4_cfg):
        for key in ('psnr', 'ssim'):
            metric = cfg['val']['metrics'][key]
            if metric['crop_border'] != 4 or metric['test_y_channel'] is not True:
                raise ValueError('E4 requires original Y-channel/crop-4 metrics.')
    rgb_opts, e4_opts = dict(rgb_cfg['network_g']), dict(e4_cfg['network_g'])
    rgb_opts.pop('type')
    e4_opts.pop('type')
    baseline, model = MambaIRv2(**rgb_opts), E4UDRMambaIRv2(**e4_opts)
    rgb_path = args.rgb_checkpoint or rgb_cfg['path']['pretrain_network_g']
    e4_path = args.e4_checkpoint or e4_cfg['path']['pretrain_network_g']
    load_weights(baseline, rgb_path)
    load_weights(model, e4_path)
    baseline, model = baseline.to(args.device).eval(), model.to(args.device).eval()
    options = dict(data_cfg['datasets']['val'])
    options.update(phase='val', scale=4)
    dataset = build_dataset(options)
    limit = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    if limit < 1 or args.visualize_count > limit:
        raise ValueError('Invalid image/visualization count.')
    output = Path(args.output).resolve()
    pictures = output/'E4_factor_maps'
    pictures.mkdir(parents=True, exist_ok=True)
    rows = []
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for index in range(limit):
        sample = dataset[index]
        rgb, depth, gt = (sample[key].unsqueeze(0) for key in ('lq', 'depth', 'gt'))
        for seed in args.seeds:
            seed_all(seed)
            rgb_sr, _, _ = infer_partitioned(baseline, rgb.to(args.device))
            seed_all(seed)
            e4_sr, maps, _ = infer_partitioned(model, rgb.to(args.device),
                                                depth.to(args.device), audit=True)
            if rgb_sr.shape != gt.shape or e4_sr.shape != gt.shape:
                raise ValueError('Prediction/GT shape mismatch.')
            rgb_error = F.avg_pool2d((rgb_sr-gt).abs().mean(1, keepdim=True),
                                      4, 4).squeeze().numpy()
            stats = map_statistics(maps, rgb_error)
            u_pearson, u_spearman = correlation(maps['ambiguity'], rgb_error)
            stats.update(uncertainty_mean=float(maps['ambiguity'].mean()),
                         uncertainty_std=float(maps['ambiguity'].std()),
                         uncertainty_rgb_error_pearson=u_pearson,
                         uncertainty_rgb_error_spearman=u_spearman,
                         confidence_mean=float(maps['confidence'].mean()),
                         confidence_std=float(maps['confidence'].std()))
            kwargs = dict(crop_border=4, test_y_channel=True)
            gt_img, rgb_img, e4_img = tensor2img(gt), tensor2img(rgb_sr), tensor2img(e4_sr)
            row = dict(image_id=Path(sample['gt_path']).stem, seed=seed, **stats,
                       baseline_psnr=float(calculate_psnr(rgb_img, gt_img, **kwargs)),
                       e4_psnr=float(calculate_psnr(e4_img, gt_img, **kwargs)),
                       baseline_ssim=float(calculate_ssim(rgb_img, gt_img, **kwargs)),
                       e4_ssim=float(calculate_ssim(e4_img, gt_img, **kwargs)))
            row['delta_psnr'] = row['e4_psnr'] - row['baseline_psnr']
            row['delta_ssim'] = row['e4_ssim'] - row['baseline_ssim']
            rows.append(row)
            if seed == args.seeds[0] and index < args.visualize_count:
                visualize(pictures/f'{index + 1:03d}_{row["image_id"]}.png',
                          sample, rgb_sr, e4_sr, maps, row['delta_psnr'])
        print(f'E4 audit {index + 1}/{limit}: {rows[-1]["image_id"]}', flush=True)
    with (output/'E4_mechanism.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows, FIELDS)
    payload = dict(experiment='E4 UDR-v2 four-factor mechanism audit',
                   uncertainty_mode=model.uncertainty_mode,
                   seeds=args.seeds, images=limit,
                   metric='uint8 Y-channel x4 crop border 4',
                   checkpoints=dict(rgb=rgb_path, e4=e4_path),
                   statistics=summary,
                   note='Descriptive mechanism audit; E4 success is decided by the separate matched-seed five-set ablation.')
    (output/'E4_statistics.json').write_text(json.dumps(payload, indent=2,
                                                         ensure_ascii=False, allow_nan=False),
                                               encoding='utf-8')
    lines = ['# E4 four-factor mechanism audit', '',
             f"{limit} DIV2K validation images × {len(args.seeds)} matched seeds.",
             '| Quantity | Mean | Valid |', '|---|---:|---:|']
    for key, item in summary.items():
        value = 'undefined' if item['mean'] is None else f"{item['mean']:.6g}"
        lines.append(f"| {key} | {value} | {item['valid']}/{item['total']} |")
    lines += ['', 'Inspect U_R, C_D, A_D and correction maps alongside RGB/E4 errors.', '']
    (output/'E4_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved E4 factor audit to', output)


def self_test():
    a = np.array([[.005, .02]], dtype=np.float32)
    maps = dict(local_alpha=a, gate=np.ones_like(a),
                correction_square_mean=np.ones_like(a) * .0001)
    row = map_statistics(maps, np.array([[0., 1.]]))
    assert row['correction_active_ratio'] == .5
    assert abs(row['correction_rms'] - .01) < 1e-7
    print('PASS: E4 factor-map statistics reuse the verified E3 alpha definitions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-config', default=str(ROOT/'options/test/mambairv2/test_UDR_RGB_reference_x4.yml'))
    parser.add_argument('--e4-config', default=str(ROOT/'options/test/mambairv2/test_E4_x4.yml'))
    parser.add_argument('--data-config', default=str(ROOT/'options/train/mambairv2/train_UDR_MambaSR_x4_phaseA.yml'))
    parser.add_argument('--rgb-checkpoint')
    parser.add_argument('--e4-checkpoint')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--seeds', nargs='+', type=int, default=[10, 11, 12])
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--visualize-count', type=int, default=20)
    parser.add_argument('--output', default=str(ROOT/'results/E4_udrv2/mechanism'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else run(args)


if __name__ == '__main__':
    main()
