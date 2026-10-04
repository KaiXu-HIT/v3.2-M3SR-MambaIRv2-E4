"""E2 uncertainty audit against fixed RGB-only reconstruction error.

Writes image/seed mechanism metrics and a Go/No-Go report. Five-set PSNR/SSIM
is run separately with evaluate_repeated.py using the E2 test YAML.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.udr.e0_mechanism_audit import (aggregate, image_statistics,
                                            infer_partitioned, load_config,
                                            load_weights, visualize)

KEYS = ('uncertainty_mean', 'uncertainty_std', 'uncertainty_p05',
        'uncertainty_p95', 'uncertainty_rgb_error_pearson',
        'uncertainty_rgb_error_spearman', 'gate_std', 'correction_rms',
        'delta_psnr', 'delta_ssim')


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_e2(model, path, kind):
    if kind == 'e2':
        load_weights(model, path)
    elif kind == 'e0':
        state = torch.load(path, map_location='cpu')
        if 'params' not in state:
            raise ValueError('E0 checkpoint lacks params.')
        weights = {(k[7:] if k.startswith('module.') else k): v
                   for k, v in state['params'].items()}
        model.load_e0_state_dict(weights)
    else:
        raise ValueError('Checkpoint kind must be e0 or e2.')


def go_decision(spearman, e0_spearman, five_set_delta, e0_five_set_delta):
    """E2 thresholds; absent measured comparisons stay pending, never assumed."""
    if spearman is None:
        return 'pending: uncertainty/error correlation undefined'
    improvement = (None if e0_spearman is None else spearman - e0_spearman)
    perf_gain = (None if five_set_delta is None or e0_five_set_delta is None
                 else five_set_delta - e0_five_set_delta)
    if spearman > .3 and perf_gain is not None and perf_gain > 0:
        return 'strong go'
    if spearman > .2 and improvement is not None and improvement >= .05:
        return 'go: correlation exceeds E0 by at least 0.05'
    if spearman <= 0 and perf_gain is not None and perf_gain <= 0:
        return 'no-go'
    return 'inconclusive: inspect maps and complete measured comparisons'


def run(args):
    from basicsr.archs.e2_udr_mambairv2_arch import E2UDRMambaIRv2
    from basicsr.archs.mambairv2_arch import MambaIRv2
    from basicsr.data import build_dataset
    from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
    from basicsr.utils.img_util import tensor2img

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Production E2 audit requires CUDA Mamba extensions.')
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use at least three distinct matched Gumbel seeds.')
    rgb_cfg = load_config(args.rgb_config)
    e2_cfg = load_config(args.e2_config)
    data_cfg = load_config(args.data_config)
    if rgb_cfg['network_g']['type'] != 'MambaIRv2' or e2_cfg['network_g']['type'] != 'E2UDRMambaIRv2':
        raise ValueError('Expected RGB reference and E2 architectures.')
    for cfg in (rgb_cfg, e2_cfg):
        for metric in ('psnr', 'ssim'):
            spec = cfg['val']['metrics'][metric]
            if spec['crop_border'] != 4 or spec['test_y_channel'] is not True:
                raise ValueError('E2 audit requires Y-channel/crop-4 metrics.')
    rgb_options, e2_options = dict(rgb_cfg['network_g']), dict(e2_cfg['network_g'])
    rgb_options.pop('type')
    e2_options.pop('type')
    baseline, e2 = MambaIRv2(**rgb_options), E2UDRMambaIRv2(**e2_options)
    rgb_path = args.rgb_checkpoint or rgb_cfg['path']['pretrain_network_g']
    e2_path = args.e2_checkpoint or e2_cfg['path']['pretrain_network_g']
    load_weights(baseline, rgb_path)
    load_e2(e2, e2_path, args.checkpoint_kind)
    baseline, e2 = baseline.to(args.device).eval(), e2.to(args.device).eval()
    data_options = dict(data_cfg['datasets']['val'])
    data_options.update(phase='val', scale=4)
    dataset = build_dataset(data_options)
    limit = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    output = Path(args.output).resolve()
    panels = output/'E2_uncertainty_maps'
    panels.mkdir(parents=True, exist_ok=True)
    rows = []
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for index in range(limit):
        sample = dataset[index]
        image_id = Path(sample['gt_path']).stem
        rgb, depth, gt = (sample[key].unsqueeze(0) for key in ('lq', 'depth', 'gt'))
        for seed in args.seeds:
            seed_all(seed)
            rgb_sr, _, _ = infer_partitioned(baseline, rgb.to(args.device))
            seed_all(seed)
            e2_sr, maps, _ = infer_partitioned(e2, rgb.to(args.device),
                                                depth.to(args.device), audit=True)
            if rgb_sr.shape != gt.shape or e2_sr.shape != gt.shape:
                raise ValueError('Prediction/GT shape mismatch; no implicit resizing.')
            error_hr = (rgb_sr - gt).abs().mean(1, keepdim=True)
            error_lr = F.avg_pool2d(error_hr, 4, 4).squeeze().numpy()
            stats = image_statistics(maps, error_lr, e2.udr.alpha.detach().item())
            gt_img = tensor2img(gt)
            baseline_img, e2_img = tensor2img(rgb_sr), tensor2img(e2_sr)
            metric = dict(crop_border=4, test_y_channel=True)
            row = dict(image_id=image_id, seed=seed,
                       uncertainty_mean=stats['ambiguity_mean'],
                       uncertainty_std=stats['ambiguity_std'],
                       uncertainty_p05=stats['ambiguity_p05'],
                       uncertainty_p95=stats['ambiguity_p95'],
                       uncertainty_rgb_error_pearson=stats['ambiguity_rgb_error_pearson'],
                       uncertainty_rgb_error_spearman=stats['ambiguity_rgb_error_spearman'],
                       gate_std=float(maps['gate'].std()),
                       correction_rms=stats['correction_rms'],
                       baseline_psnr=float(calculate_psnr(baseline_img, gt_img, **metric)),
                       e2_psnr=float(calculate_psnr(e2_img, gt_img, **metric)),
                       baseline_ssim=float(calculate_ssim(baseline_img, gt_img, **metric)),
                       e2_ssim=float(calculate_ssim(e2_img, gt_img, **metric)))
            row['delta_psnr'] = row['e2_psnr'] - row['baseline_psnr']
            row['delta_ssim'] = row['e2_ssim'] - row['baseline_ssim']
            rows.append(row)
            if seed == args.seeds[0] and index < args.visualize_count:
                visualize(panels/f'{index + 1:03d}_{image_id}.png', sample,
                          rgb_sr, e2_sr, error_hr.squeeze().numpy(), maps)
        print(f'E2 audit {index + 1}/{limit}: {image_id}', flush=True)
    with (output/'E2_mechanism.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows, KEYS)
    e0_rho = None
    if args.e0_statistics:
        old = json.loads(Path(args.e0_statistics).read_text(encoding='utf-8'))
        e0_rho = old['statistics']['ambiguity_rgb_error_spearman']['mean']
    five_delta = e0_delta = None
    if args.e2_five_set and args.e0_five_set:
        current = json.loads(Path(args.e2_five_set).read_text(encoding='utf-8'))
        previous = json.loads(Path(args.e0_five_set).read_text(encoding='utf-8'))
        five_delta = current['average_delta_vs_rgb']['udr']
        e0_delta = previous['average_delta_vs_rgb']['udr']
    rho = summary['uncertainty_rgb_error_spearman']['mean']
    decision = go_decision(rho, e0_rho, five_delta, e0_delta)
    payload = dict(experiment='E2 uncertainty-only mechanism audit',
                   variant=e2.uncertainty_mode, seeds=args.seeds, images=limit,
                   training='none in this audit', checkpoint_kind=args.checkpoint_kind,
                   checkpoints=dict(rgb=rgb_path, e2=e2_path),
                   metric='uint8 Y-channel, crop border 4, x4',
                   rgb_error='channel-mean absolute fixed RGB-only SR error, x4 area pooled to LR',
                   statistics=summary, e0_uncertainty_spearman=e0_rho,
                   e2_five_set_delta=five_delta, e0_five_set_delta=e0_delta,
                   decision=decision, go_convention='clear improvement means +0.05 Spearman over E0')
    (output/'E2_statistics.json').write_text(json.dumps(payload, indent=2,
                                                         ensure_ascii=False, allow_nan=False),
                                               encoding='utf-8')
    lines = [f'# E2 {e2.uncertainty_mode} mechanism audit', '',
             f'{limit} DIV2K validation images × {len(args.seeds)} matched seeds.',
             'Fixed RGB-only error; Y-channel/crop-4 PSNR and SSIM.', '',
             '| Quantity | Mean | Valid rows |', '|---|---:|---:|']
    for key, item in summary.items():
        value = 'undefined' if item['mean'] is None else f"{item['mean']:.6g}"
        lines.append(f"| {key} | {value} | {item['valid']}/{item['total']} |")
    lines += ['', f'Decision: **{decision}**.',
              'The decision remains pending if E0 audit or five-set comparison was not supplied.',
              'Correlations are descriptive; inspect the maps before causal claims.', '']
    (output/'E2_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved E2 audit to', output)


def self_test():
    assert go_decision(.35, .1, .08, .04) == 'strong go'
    assert go_decision(.25, .15, None, None).startswith('go:')
    assert go_decision(-.01, .1, -.03, 0) == 'no-go'
    assert go_decision(.1, None, None, None).startswith('inconclusive')
    print('PASS: E2 Go/No-Go gates preserve missing-data state')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-config', default=str(ROOT/'options/test/mambairv2/test_UDR_RGB_reference_x4.yml'))
    parser.add_argument('--e2-config', default=str(ROOT/'options/test/mambairv2/test_E2_U3_x4.yml'))
    parser.add_argument('--data-config', default=str(ROOT/'options/train/mambairv2/train_E2_U3_phaseA_x4.yml'))
    parser.add_argument('--rgb-checkpoint')
    parser.add_argument('--e2-checkpoint')
    parser.add_argument('--checkpoint-kind', choices=('e0', 'e2'), default='e2')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--seeds', nargs='+', type=int, default=[10, 11, 12])
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--visualize-count', type=int, default=20)
    parser.add_argument('--e0-statistics')
    parser.add_argument('--e2-five-set')
    parser.add_argument('--e0-five-set')
    parser.add_argument('--output', default=str(ROOT/'results/E2_udrv2_uncertainty/U3_audit'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else run(args)


if __name__ == '__main__':
    main()
