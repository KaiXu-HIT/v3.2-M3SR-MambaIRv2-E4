"""E3: matched-seed spatial-alpha audit against E0 and fixed RGB baseline."""
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

FIELDS = ('local_alpha_mean', 'local_alpha_std', 'local_alpha_p05',
          'local_alpha_p50', 'local_alpha_p95', 'local_alpha_max',
          'correction_rms', 'correction_active_ratio', 'gate_std',
          'local_alpha_rgb_error_pearson', 'local_alpha_rgb_error_spearman',
          'delta_psnr', 'delta_ssim')


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_statistics(maps, rgb_error):
    alpha = maps['local_alpha']
    if alpha.shape != rgb_error.shape:
        raise ValueError('LR alpha and RGB-only reconstruction error do not align.')
    pearson, spearman = correlation(alpha, rgb_error)
    return dict(local_alpha_mean=float(alpha.mean()),
                local_alpha_std=float(alpha.std()),
                local_alpha_p05=float(np.percentile(alpha, 5)),
                local_alpha_p50=float(np.percentile(alpha, 50)),
                local_alpha_p95=float(np.percentile(alpha, 95)),
                local_alpha_max=float(alpha.max()),
                correction_rms=float(np.sqrt(maps['correction_square_mean'].mean())),
                correction_active_ratio=float((alpha > .01).mean()),
                gate_std=float(maps['gate'].std()),
                local_alpha_rgb_error_pearson=pearson,
                local_alpha_rgb_error_spearman=spearman)


def _rgb(tensor):
    return tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)


def visualize(path, sample, rgb_sr, e3_sr, maps, delta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gt = sample['gt'].unsqueeze(0)
    panels = [
        ('LR RGB', _rgb(sample['lq'].unsqueeze(0)), None),
        ('LR Depth', sample['depth'].squeeze().numpy(), 'viridis'),
        ('GT', _rgb(gt), None),
        ('RGB-only SR', _rgb(rgb_sr), None),
        ('E3 SR', _rgb(e3_sr), None),
        ('RGB |SR-GT|', (rgb_sr - gt).abs().mean(1).squeeze().numpy(), 'magma'),
        ('E3 |SR-GT|', (e3_sr - gt).abs().mean(1).squeeze().numpy(), 'magma'),
        ('E0 routing uncertainty', maps['ambiguity'], 'viridis'),
        ('Depth confidence', maps['confidence'], 'viridis'),
        ('Gate', maps['gate'], 'viridis'),
        ('Local alpha', maps['local_alpha'], 'viridis'),
        ('Feature correction', maps['correction_abs'], 'magma')]
    fig, axes = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    for ax, (label, value, cmap) in zip(axes.flat, panels):
        ax.imshow(value, cmap=cmap, vmin=0 if cmap else None,
                  vmax=(.1 if label == 'Local alpha' else
                        1 if cmap == 'viridis' else
                        float(np.percentile(value, 99)) + 1e-12 if cmap else None))
        ax.set_title(label, fontsize=9)
        ax.axis('off')
    fig.suptitle(f'E3 ΔPSNR {delta:+.4f} dB (representative seed)')
    fig.savefig(path, dpi=125)
    plt.close(fig)


def measured_comparison(args):
    if not args.e0_five_set and not args.e3_five_set:
        return None
    if not args.e0_five_set or not args.e3_five_set:
        raise ValueError('Provide both E0 and E3 five-set summaries.')
    old = json.loads(Path(args.e0_five_set).read_text(encoding='utf-8'))
    new = json.loads(Path(args.e3_five_set).read_text(encoding='utf-8'))
    return dict(e0_average_delta=old['average_delta_vs_rgb']['udr'],
                e3_average_delta=new['average_delta_vs_rgb']['udr'],
                average_gain_vs_e0=(new['average_delta_vs_rgb']['udr']
                                    - old['average_delta_vs_rgb']['udr']),
                e0_urban_delta=old['summary']['Urban100']['udr']['delta_psnr_vs_rgb'],
                e3_urban_delta=new['summary']['Urban100']['udr']['delta_psnr_vs_rgb'])


def run(args):
    from basicsr.archs.e3_local_alpha_arch import E3LocalAlphaMambaIRv2
    from basicsr.archs.mambairv2_arch import MambaIRv2
    from basicsr.data import build_dataset
    from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
    from basicsr.utils.img_util import tensor2img

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Production E3 audit needs CUDA and Mamba CUDA extensions.')
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use at least three distinct matched seeds.')
    rgb_cfg, e3_cfg = load_config(args.rgb_config), load_config(args.e3_config)
    data_cfg = load_config(args.data_config)
    if rgb_cfg['network_g']['type'] != 'MambaIRv2' or e3_cfg['network_g']['type'] != 'E3LocalAlphaMambaIRv2':
        raise ValueError('Expected fixed RGB baseline and E3 architecture.')
    for cfg in (rgb_cfg, e3_cfg):
        for name in ('psnr', 'ssim'):
            metric = cfg['val']['metrics'][name]
            if metric['crop_border'] != 4 or metric['test_y_channel'] is not True:
                raise ValueError('E3 requires the original Y-channel/crop-4 protocol.')
    rgb_kw, e3_kw = dict(rgb_cfg['network_g']), dict(e3_cfg['network_g'])
    rgb_kw.pop('type')
    e3_kw.pop('type')
    baseline, model = MambaIRv2(**rgb_kw), E3LocalAlphaMambaIRv2(**e3_kw)
    rgb_path = args.rgb_checkpoint or rgb_cfg['path']['pretrain_network_g']
    e3_path = args.e3_checkpoint or e3_cfg['path']['pretrain_network_g']
    load_weights(baseline, rgb_path)
    load_weights(model, e3_path)
    baseline, model = baseline.to(args.device).eval(), model.to(args.device).eval()
    options = dict(data_cfg['datasets']['val'])
    options.update(phase='val', scale=4)
    dataset = build_dataset(options)
    limit = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    if limit < 1 or args.visualize_count > limit:
        raise ValueError('Requested image/visualization count is invalid.')
    output = Path(args.output).resolve()
    pictures = output/'E3_local_alpha_maps'
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
            e3_sr, maps, _ = infer_partitioned(model, rgb.to(args.device),
                                                depth.to(args.device), audit=True)
            if rgb_sr.shape != gt.shape or e3_sr.shape != gt.shape:
                raise ValueError('Prediction and GT shapes differ.')
            error_lr = F.avg_pool2d((rgb_sr - gt).abs().mean(1, keepdim=True), 4, 4).squeeze().numpy()
            stats = map_statistics(maps, error_lr)
            metric = dict(crop_border=4, test_y_channel=True)
            gt_img = tensor2img(gt)
            base_img, e3_img = tensor2img(rgb_sr), tensor2img(e3_sr)
            row = dict(image_id=Path(sample['gt_path']).stem, seed=seed, **stats,
                       baseline_psnr=float(calculate_psnr(base_img, gt_img, **metric)),
                       e3_psnr=float(calculate_psnr(e3_img, gt_img, **metric)),
                       baseline_ssim=float(calculate_ssim(base_img, gt_img, **metric)),
                       e3_ssim=float(calculate_ssim(e3_img, gt_img, **metric)))
            row['delta_psnr'] = row['e3_psnr'] - row['baseline_psnr']
            row['delta_ssim'] = row['e3_ssim'] - row['baseline_ssim']
            rows.append(row)
            if seed == args.seeds[0] and index < args.visualize_count:
                visualize(pictures/f'{index + 1:03d}_{row["image_id"]}.png',
                          sample, rgb_sr, e3_sr, maps, row['delta_psnr'])
        print(f'E3 audit {index + 1}/{limit}: {rows[-1]["image_id"]}', flush=True)
    with (output/'E3_local_alpha.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stats = aggregate(rows, FIELDS)
    performance = measured_comparison(args)
    span = (stats['local_alpha_p95']['mean'] - stats['local_alpha_p05']['mean'])
    report = dict(experiment='E3 local correction strength', seeds=args.seeds,
                  dataset=options['name'], images=limit, checkpoint=e3_path,
                  rgb_checkpoint=rgb_path,
                  metric='uint8 Y-channel x4 crop border 4',
                  active_definition='local alpha > 0.01',
                  statistics=stats, mean_p95_minus_p05=span,
                  five_set_comparison=performance,
                  interpretation='Local alpha has spatial variation if std and P95-P05 exceed zero; inspect whether stronger corrections are located at useful regions. Correlation is descriptive.')
    (output/'E3_statistics.json').write_text(json.dumps(report, indent=2,
                                                         ensure_ascii=False, allow_nan=False),
                                               encoding='utf-8')
    lines = ['# E3 local alpha audit', '',
             f"DIV2K validation: {limit} images × {len(args.seeds)} matched seeds.",
             'E3 changes alpha only; uncertainty remains E0 routing entropy.',
             'PSNR/SSIM: original uint8 Y channel, x4, crop border 4.', '',
             '| Quantity | Mean | Valid |', '|---|---:|---:|']
    for key, item in stats.items():
        value = 'undefined' if item['mean'] is None else f"{item['mean']:.6g}"
        lines.append(f"| {key} | {value} | {item['valid']}/{item['total']} |")
    lines += ['', f'Mean P95−P05 alpha span: {span:.6g}.',
              'Global E0 alpha has zero within-image spatial standard deviation by definition.',
              'Review alpha/gate/correction maps before attributing performance to local selectivity.', '']
    if performance:
        lines.append('Five-set comparison: ' + json.dumps(performance, ensure_ascii=False))
    else:
        lines.append('Five-set comparison pending: supply both measured summary.json files.')
    (output/'E3_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Saved E3 audit to', output)


def self_test():
    a = np.array([[0., .005, .01, .02]], dtype=np.float32)
    maps = dict(local_alpha=a, gate=np.ones_like(a),
                correction_square_mean=np.ones_like(a) * .0004)
    row = map_statistics(maps, np.array([[0., 1., 2., 3.]]))
    assert row['correction_active_ratio'] == .25
    assert abs(row['correction_rms'] - .02) < 1e-7
    assert row['local_alpha_rgb_error_spearman'] == 1.
    print('PASS: local alpha quantiles, active ratio, correction RMS and error correlation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-config', default=str(ROOT/'options/test/mambairv2/test_UDR_RGB_reference_x4.yml'))
    parser.add_argument('--e3-config', default=str(ROOT/'options/test/mambairv2/test_E3_a1e4_x4.yml'))
    parser.add_argument('--data-config', default=str(ROOT/'options/train/mambairv2/train_E3_a1e4_phaseA_x4.yml'))
    parser.add_argument('--rgb-checkpoint')
    parser.add_argument('--e3-checkpoint')
    parser.add_argument('--e0-five-set')
    parser.add_argument('--e3-five-set')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=[10, 11, 12])
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--visualize-count', type=int, default=20)
    parser.add_argument('--output', default=str(ROOT/'results/E3_local_alpha/a1e4_audit'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else run(args)


if __name__ == '__main__':
    main()
