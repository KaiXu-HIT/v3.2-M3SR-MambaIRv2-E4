"""E1: image-level Manga109/Urban100 diagnosis of the frozen E0 checkpoints.

No training occurs here. Run with --self-test without CUDA/data/checkpoints; see
docs/E1_GUIDE.md for the full matched-seed evaluation command and output contract.
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
from scripts.udr.e0_mechanism_audit import infer_partitioned, load_config, load_weights

DATASETS = ('Manga109', 'Urban100')
FIELDS = ('baseline_psnr', 'udr_psnr', 'delta_psnr', 'baseline_ssim',
          'udr_ssim', 'delta_ssim', 'depth_edge_density', 'rgb_edge_density',
          'depth_rgb_edge_alignment', 'unmatched_depth_edge_fraction',
          'repetition_score', 'mean_confidence', 'mean_uncertainty',
          'mean_gate', 'mean_correction')
CORRELATES = ('depth_rgb_edge_alignment', 'mean_confidence',
              'mean_correction', 'depth_edge_density', 'repetition_score',
              'unmatched_depth_edge_fraction')


def sobel_magnitude(gray):
    """Use exactly the production UDR /8 Sobel kernel and replicate padding."""
    x = torch.as_tensor(gray, dtype=torch.float32).reshape(1, 1, *gray.shape)
    k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
    kernels = torch.stack((k, k.t())).unsqueeze(1)
    return torch.linalg.vector_norm(F.conv2d(F.pad(x, (1, 1, 1, 1),
                                                     mode='replicate'), kernels),
                                    dim=1).squeeze().numpy()


def repetition_score(edge):
    """Maximum horizontal/vertical edge autocorrelation at 4..16 LR pixels.

    This is a repeat-texture proxy, not a semantic building/geometry classifier.
    Constant or very small images have undefined autocorrelation (null).
    """
    h, w = edge.shape
    values = []
    for shift in range(4, min(16, max(h, w) // 2) + 1):
        for axis in (0, 1):
            if edge.shape[axis] <= 2 * shift:
                continue
            a = edge[:-shift, :] if axis == 0 else edge[:, :-shift]
            b = edge[shift:, :] if axis == 0 else edge[:, shift:]
            a, b = a.astype(np.float64), b.astype(np.float64)
            a -= a.mean()
            b -= b.mean()
            denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
            if denom > 1e-10:
                values.append(float(np.sum(a * b) / denom))
    return max(values) if values else None


def structure_metrics(rgb, depth, rgb_threshold, depth_threshold):
    """Fixed thresholds make edge density comparable between images.

    Alignment follows E1's RGB-edge-normalized formula; when there are no RGB
    edges, leave it null rather than manufacturing a perfect agreement score.
    """
    rgb_np = rgb.squeeze(0).detach().cpu().numpy()
    depth_np = depth.squeeze().detach().cpu().numpy()
    luma = np.tensordot(np.array([.299, .587, .114], dtype=np.float32),
                        rgb_np, axes=(0, 0))
    rgb_edge = sobel_magnitude(luma)
    depth_edge = sobel_magnitude(depth_np)
    rmask, dmask = rgb_edge > rgb_threshold, depth_edge > depth_threshold
    matched = np.logical_and(rmask, dmask)
    return dict(rgb_edge_density=float(rmask.mean()),
                depth_edge_density=float(dmask.mean()),
                depth_rgb_edge_alignment=(float(matched.sum() / rmask.sum())
                                          if rmask.any() else None),
                unmatched_depth_edge_fraction=(float(np.logical_and(dmask, ~rmask).sum()
                                                     / dmask.sum()) if dmask.any() else None),
                repetition_score=repetition_score(rgb_edge)), rgb_edge, depth_edge


def safe_correlation(rows, x_key, y_key='delta_psnr'):
    """Pearson/Spearman with pairwise-null exclusion and no fake zero values."""
    from scipy.stats import pearsonr, spearmanr
    pairs = [(row[x_key], row[y_key]) for row in rows
             if row.get(x_key) is not None and row.get(y_key) is not None
             and math.isfinite(float(row[x_key])) and math.isfinite(float(row[y_key]))]
    if len(pairs) < 3:
        return dict(pearson=None, spearman=None, valid=len(pairs))
    values = np.asarray(pairs, dtype=np.float64)
    if np.std(values[:, 0]) < 1e-10 or np.std(values[:, 1]) < 1e-10:
        return dict(pearson=None, spearman=None, valid=len(pairs))
    return dict(pearson=float(pearsonr(values[:, 0], values[:, 1])[0]),
                spearman=float(spearmanr(values[:, 0], values[:, 1])[0]),
                valid=len(pairs))


def select_groups(rows, count=20):
    """Disjoint top, bottom, and central 20 ranked by mean matched-seed gain."""
    if len(rows) < 3 * count:
        raise ValueError('E1 requires at least 60 images per dataset for disjoint groups.')
    ranked = sorted(rows, key=lambda row: (row['delta_psnr'], row['image_id']))
    first_middle = (len(ranked) - count) // 2
    return dict(top20=list(reversed(ranked[-count:])),
                bottom20=ranked[:count],
                median20=ranked[first_middle:first_middle + count])


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def aggregate_seeds(rows):
    """One image row is the average of its matched-seed rows, not a seed pick."""
    result = dict(dataset=rows[0]['dataset'], image_id=rows[0]['image_id'],
                  seeds=len(rows))
    for field in FIELDS:
        result[field] = _mean([row[field] for row in rows])
    for field in ('baseline_psnr', 'udr_psnr', 'delta_psnr', 'baseline_ssim',
                  'udr_ssim', 'delta_ssim'):
        result[field + '_std'] = (float(np.std([row[field] for row in rows], ddof=1))
                                   if len(rows) > 1 else 0.0)
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refusing to write an empty E1 table.')
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed):
    # RGB and UDR have hard Gumbel routing even in eval; reset before each run.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_seed(sample, baseline, udr, seed, device, metric_functions,
                  structure):
    calculate_psnr, calculate_ssim, tensor2img = metric_functions
    rgb = sample['lq'].unsqueeze(0).to(device)
    depth = sample['depth'].unsqueeze(0).to(device)
    gt = sample['gt'].unsqueeze(0)
    seed_all(seed)
    rgb_sr, _, _ = infer_partitioned(baseline, rgb)
    seed_all(seed)
    udr_sr, maps, _ = infer_partitioned(udr, rgb, depth, audit=True)
    if rgb_sr.shape != gt.shape or udr_sr.shape != gt.shape:
        raise ValueError('SR output and GT differ; E1 never resizes predictions.')
    gt_img = tensor2img(gt)
    rgb_img, udr_img = tensor2img(rgb_sr), tensor2img(udr_sr)
    # Match the original test YAML: uint8 Y-channel, four-pixel HR crop.
    kwargs = dict(crop_border=4, test_y_channel=True)
    row = dict(seed=seed, baseline_psnr=float(calculate_psnr(rgb_img, gt_img, **kwargs)),
               udr_psnr=float(calculate_psnr(udr_img, gt_img, **kwargs)),
               baseline_ssim=float(calculate_ssim(rgb_img, gt_img, **kwargs)),
               udr_ssim=float(calculate_ssim(udr_img, gt_img, **kwargs)),
               mean_confidence=float(maps['confidence'].mean()),
               mean_uncertainty=float(maps['ambiguity'].mean()),
               mean_gate=float(maps['gate'].mean()),
               mean_correction=float(maps['correction_abs'].mean()), **structure)
    row['delta_psnr'] = row['udr_psnr'] - row['baseline_psnr']
    row['delta_ssim'] = row['udr_ssim'] - row['baseline_ssim']
    return row, (rgb_sr, udr_sr, maps)


def _rgb(tensor):
    return tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)


def visualize(path, sample, rgb_sr, udr_sr, maps, rgb_edge, depth_edge, row):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gt = sample['gt'].unsqueeze(0)
    baseline_error = (rgb_sr - gt).abs().mean(1).squeeze().numpy()
    udr_error = (udr_sr - gt).abs().mean(1).squeeze().numpy()
    panels = [
        ('LR RGB', _rgb(sample['lq'].unsqueeze(0)), None),
        ('LR normalized Depth', sample['depth'].squeeze().numpy(), 'viridis'),
        ('RGB Sobel', rgb_edge, 'magma'), ('Depth Sobel', depth_edge, 'magma'),
        ('GT', _rgb(gt), None), ('RGB baseline', _rgb(rgb_sr), None),
        ('UDR', _rgb(udr_sr), None), ('|RGB error|', baseline_error, 'magma'),
        ('|UDR error|', udr_error, 'magma'),
        ('Routing uncertainty', maps['ambiguity'], 'viridis'),
        ('Depth confidence', maps['confidence'], 'viridis'),
        ('Gate', maps['gate'], 'viridis'),
        ('Feature correction', maps['correction_abs'], 'magma')]
    fig, axes = plt.subplots(4, 4, figsize=(16, 14), constrained_layout=True)
    for ax, (label, image, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap, vmin=0 if cmap else None,
                  vmax=(1 if cmap == 'viridis' else
                        float(np.percentile(image, 99)) + 1e-12 if cmap else None))
        ax.set_title(label, fontsize=9)
        ax.axis('off')
    for ax in axes.flat[len(panels):]:
        ax.axis('off')
    fig.suptitle(f"{row['dataset']} / {row['image_id']} / mean ΔPSNR "
                 f"{row['delta_psnr']:+.4f} dB / seed {row['visual_seed']}")
    fig.savefig(path, dpi=125)
    plt.close(fig)


def verify_configs(rgb_config, udr_config):
    if rgb_config['network_g']['type'] != 'MambaIRv2':
        raise ValueError('RGB reference architecture changed.')
    if udr_config['network_g']['type'] != 'UDRMambaIRv2':
        raise ValueError('UDR architecture changed.')
    for config in (rgb_config, udr_config):
        for metric in ('psnr', 'ssim'):
            spec = config['val']['metrics'][metric]
            if spec['crop_border'] != 4 or spec['test_y_channel'] is not True:
                raise ValueError('E1 requires original Y-channel x4 metric protocol.')
    for dataset in DATASETS:
        rgb_data = next(v for v in rgb_config['datasets'].values() if v['name'] == dataset)
        udr_data = next(v for v in udr_config['datasets'].values() if v['name'] == dataset)
        for key in ('dataroot_gt', 'dataroot_lq', 'filename_tmpl'):
            if rgb_data[key] != udr_data[key]:
                raise ValueError(f'{dataset}: RGB and UDR {key} differ.')
    return True


def build_models(args, rgb_config, udr_config):
    from basicsr.archs.mambairv2_arch import MambaIRv2
    from basicsr.archs.udr_mambairv2_arch import UDRMambaIRv2
    rgb_options = dict(rgb_config['network_g'])
    udr_options = dict(udr_config['network_g'])
    rgb_options.pop('type')
    udr_options.pop('type')
    baseline, udr = MambaIRv2(**rgb_options), UDRMambaIRv2(**udr_options)
    rgb_path = args.rgb_checkpoint or rgb_config['path']['pretrain_network_g']
    udr_path = args.udr_checkpoint or udr_config['path']['pretrain_network_g']
    load_weights(baseline, rgb_path)
    load_weights(udr, udr_path)
    return baseline.to(args.device).eval(), udr.to(args.device).eval(), rgb_path, udr_path


def run(args):
    from basicsr.data import build_dataset
    from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
    from basicsr.utils.img_util import tensor2img
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Production E1 requires CUDA and Mamba CUDA extensions.')
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('E1 requires at least three distinct matched seeds.')
    if args.rgb_edge_threshold <= 0 or args.depth_edge_threshold <= 0:
        raise ValueError('Sobel edge thresholds must be positive.')
    rgb_config, udr_config = load_config(args.rgb_config), load_config(args.udr_config)
    verify_configs(rgb_config, udr_config)
    baseline, udr, rgb_path, udr_path = build_models(args, rgb_config, udr_config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    metric_functions = calculate_psnr, calculate_ssim, tensor2img
    per_seed, per_image, selections, datasets = [], [], {}, {}
    for name in DATASETS:
        data_options = next(dict(v) for v in udr_config['datasets'].values()
                            if v['name'] == name)
        data_options.update(phase='val', scale=4)
        dataset = build_dataset(data_options)
        if len(dataset) < 60:
            raise ValueError(f'{name} has fewer than 60 aligned images.')
        datasets[name] = dataset
        for index in range(len(dataset)):
            sample = dataset[index]
            image_id = Path(sample['gt_path']).stem
            # Image structure is seed-independent; inspect the same full LR pair
            # that the E0 checkpoint consumes, without changing dataset paths.
            structure, _, _ = structure_metrics(sample['lq'].unsqueeze(0),
                                                   sample['depth'].unsqueeze(0),
                                                   args.rgb_edge_threshold,
                                                   args.depth_edge_threshold)
            seed_rows = []
            for seed in args.seeds:
                values, _ = evaluate_seed(sample, baseline, udr, seed,
                                          args.device, metric_functions, structure)
                row = dict(dataset=name, image_id=image_id, **values)
                seed_rows.append(row)
                per_seed.append(row)
            per_image.append(aggregate_seeds(seed_rows))
            print(f'E1 {name} {index + 1}/{len(dataset)}: {image_id}', flush=True)
        dataset_rows = [row for row in per_image if row['dataset'] == name]
        selections[name] = select_groups(dataset_rows)
    write_csv(output/'E1_per_seed.csv', per_seed)
    write_csv(output/'E1_per_image.csv', per_image)

    correlations = {}
    for name in DATASETS:
        dataset_rows = [row for row in per_image if row['dataset'] == name]
        correlations[name] = {
            'images': len(dataset_rows),
            'mean_delta_psnr': _mean([row['delta_psnr'] for row in dataset_rows]),
            'mean_delta_ssim': _mean([row['delta_ssim'] for row in dataset_rows]),
            'positive_delta_images': sum(row['delta_psnr'] > 0 for row in dataset_rows),
            'correlations_with_delta_psnr': {
                field: safe_correlation(dataset_rows, field) for field in CORRELATES},
            'groups': {group: [row['image_id'] for row in selected]
                       for group, selected in selections[name].items()}}
    metadata = dict(experiment='E1 Manga109 versus Urban100 image-level diagnosis',
                    training='none', seeds=args.seeds,
                    checkpoints=dict(rgb=rgb_path, udr=udr_path),
                    performance='uint8 Y-channel PSNR/SSIM, crop border 4, x4',
                    edge_thresholds=dict(rgb=args.rgb_edge_threshold,
                                         depth=args.depth_edge_threshold),
                    edge_formula='matched RGB/depth edge pixels divided by RGB edge pixels',
                    repetition='maximum edge autocorrelation, horizontal/vertical lag 4..16 LR pixels',
                    datasets=correlations)
    (output/'E1_correlation.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False),
        encoding='utf-8')

    # Re-infer only the selected 120 images for panels; ranking used all seeds.
    # Save the first-seed representative and label it so it is not mistaken for
    # the across-seed mean used in the CSV and correlations.
    for name in DATASETS:
        dataset = datasets[name]
        indices = {Path(pair['gt_path']).stem: i for i, pair in enumerate(dataset.paths)}
        for group, rows in selections[name].items():
            directory = output/f"E1_{group}_{'manga' if name == 'Manga109' else 'urban'}"
            directory.mkdir(exist_ok=True)
            for rank, row in enumerate(rows, 1):
                sample = dataset[indices[row['image_id']]]
                _, rgb_edge, depth_edge = structure_metrics(
                    sample['lq'].unsqueeze(0), sample['depth'].unsqueeze(0),
                    args.rgb_edge_threshold, args.depth_edge_threshold)
                _, (rgb_sr, udr_sr, maps) = evaluate_seed(
                    sample, baseline, udr, args.seeds[0], args.device,
                    metric_functions, dict(rgb_edge_density=row['rgb_edge_density'],
                                           depth_edge_density=row['depth_edge_density'],
                                           depth_rgb_edge_alignment=row['depth_rgb_edge_alignment'],
                                           unmatched_depth_edge_fraction=row['unmatched_depth_edge_fraction'],
                                           repetition_score=row['repetition_score']))
                visualize(directory/f'{rank:02d}_{row["image_id"]}.png', sample,
                          rgb_sr, udr_sr, maps, rgb_edge, depth_edge,
                          dict(row, visual_seed=args.seeds[0]))
                print(f'E1 panel {name} {group} {rank}/20', flush=True)

    lines = ['# E1 Manga109 / Urban100 diagnosis', '',
             'No training. The original E0 RGB and UDR checkpoints are compared '
             f'with matched Gumbel seeds {args.seeds}.',
             'Per-image values and ranking are means across seeds; panels use '
             f'representative seed {args.seeds[0]}.',
             'PSNR/SSIM: original uint8 Y channel, HR crop border 4.',
             f"Edge thresholds on /8 Sobel magnitude: RGB {args.rgb_edge_threshold}, "
             f"Depth {args.depth_edge_threshold}.",
             'Repetition score is a simple edge periodicity proxy, not a semantic label.', '']
    for name in DATASETS:
        result = correlations[name]
        lines += [f'## {name}', '',
                  f"Images: {result['images']}; positive ΔPSNR: "
                  f"{result['positive_delta_images']}; mean ΔPSNR: "
                  f"{result['mean_delta_psnr']:+.6f} dB; mean ΔSSIM: "
                  f"{result['mean_delta_ssim']:+.6f}.", '',
                  '| ΔPSNR versus | Pearson | Spearman | valid images |',
                  '|---|---:|---:|---:|']
        for field, item in result['correlations_with_delta_psnr'].items():
            p = 'undefined' if item['pearson'] is None else f"{item['pearson']:+.4f}"
            s = 'undefined' if item['spearman'] is None else f"{item['spearman']:+.4f}"
            lines.append(f"| {field} | {p} | {s} | {item['valid']} |")
        lines += ['', 'Inspect top20/bottom20/median20 panels against both SR errors, '
                  'Depth edges, confidence, gate and correction before attributing '
                  'any degradation to pseudo-depth or repeated texture.', '']
    lines += ['## Interpretation rules', '',
              'A positive edge-alignment/ΔPSNR association supports testing local '
              'geometry-aligned gating in later experiments; it does not prove causation.',
              'High Depth-edge density plus low alignment in negative examples supports '
              'the hypothesis of texture-induced pseudo-depth edges only after panel review.',
              'Correlations are descriptive and are undefined for constant variables.', '',
              'Outputs: E1_per_image.csv, E1_per_seed.csv, E1_correlation.json, '
              'E1_summary.md, and six selected-image panel directories.', '']
    (output/'E1_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved E1 outputs to', output)


def self_test():
    import tempfile
    y, x = np.indices((64, 64))
    periodic = ((x // 4) % 2).astype(np.float32)
    random = np.random.RandomState(13).rand(64, 64).astype(np.float32)
    assert repetition_score(periodic) > repetition_score(random)
    rgb = torch.from_numpy(np.stack((periodic,) * 3))[None]
    depth = torch.from_numpy(periodic)[None, None]
    structure, _, _ = structure_metrics(rgb, depth, .05, .05)
    assert structure['depth_rgb_edge_alignment'] == 1
    assert structure['unmatched_depth_edge_fraction'] == 0
    assert structure['rgb_edge_density'] > 0
    rows = [dict(image_id=f'{i:03d}', delta_psnr=float(i), dataset='x')
            for i in range(100)]
    groups = select_groups(rows)
    assert all(len(items) == 20 for items in groups.values())
    assert len({r['image_id'] for items in groups.values() for r in items}) == 60
    assert safe_correlation(rows, 'delta_psnr')['pearson'] > .999
    with tempfile.TemporaryDirectory() as folder:
        write_csv(Path(folder)/'test.csv', rows)
        assert len((Path(folder)/'test.csv').read_text(encoding='utf-8-sig').splitlines()) == 101
    print('PASS: aligned Sobel edges, periodicity, correlations, disjoint groups, CSV')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-config', default=str(ROOT/'options/test/mambairv2/test_UDR_RGB_reference_x4.yml'))
    parser.add_argument('--udr-config', default=str(ROOT/'options/test/mambairv2/test_UDR_MambaSR_x4.yml'))
    parser.add_argument('--rgb-checkpoint', help='Fixed RGB baseline net_g_490000.pth override.')
    parser.add_argument('--udr-checkpoint', help='Trained E0 Phase B net_g_100000.pth override.')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=[10, 11, 12])
    parser.add_argument('--rgb-edge-threshold', type=float, default=.05)
    parser.add_argument('--depth-edge-threshold', type=float, default=.05)
    parser.add_argument('--output', default=str(ROOT/'results/E1_dataset_diagnosis'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else run(args)


if __name__ == '__main__':
    main()
