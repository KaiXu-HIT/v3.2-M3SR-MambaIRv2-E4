"""E0: read-only audit of a trained v3.2 UDR and the fixed RGB baseline.

No weights, training options, or data paths are modified. The two networks use
the same image partitioning and per-image Gumbel seed. See docs/E0_GUIDE.md.
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
from scipy.stats import pearsonr, spearmanr
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MAP_KEYS = ('route_entropy', 'route_maxprob', 'route_margin', 'ambiguity',
            'confidence', 'gate', 'depth_gradient', 'depth_residual_abs',
            'correction_abs', 'correction_square_mean', 'correction_channel_max',
            'feature_abs')
CORRELATION_KEYS = ('ambiguity', 'confidence', 'gate', 'depth_residual_abs')
LOG_KEYS = ('route_entropy_mean', 'route_entropy_std', 'route_maxprob_mean',
            'route_margin_mean', 'ambiguity_mean', 'ambiguity_std',
            'ambiguity_p05', 'ambiguity_p25', 'ambiguity_p50',
            'ambiguity_p75', 'ambiguity_p95', 'confidence_mean',
            'confidence_std', 'gate_mean', 'gate_std', 'alpha',
            'correction_rms', 'correction_abs_mean', 'correction_abs_max')


def partition_plan(height, width):
    """Exact borders/overlap from MambaIRv2Model.test and UDRMambaIRv2Model.test."""
    count_h, count_w = height // 200 + 1, width // 200 + 1
    pad_h = (-height) % count_h
    pad_w = (-width) % count_w
    padded_h, padded_w = height + pad_h, width + pad_w
    split_h, split_w = padded_h // count_h, padded_w // count_w
    shave_h, shave_w = split_h // 10, split_w // 10
    plan = []
    for i in range(count_h):
        for j in range(count_w):
            y0 = i * split_h - (shave_h if i else 0)
            y1 = (i + 1) * split_h + (shave_h if i < count_h - 1 else 0)
            x0 = j * split_w - (shave_w if j else 0)
            x1 = (j + 1) * split_w + (shave_w if j < count_w - 1 else 0)
            source = (slice(y0, y1), slice(x0, x1))
            target = (slice(i * split_h, (i + 1) * split_h),
                      slice(j * split_w, (j + 1) * split_w))
            inner = (slice(shave_h if i else 0, (shave_h if i else 0) + split_h),
                     slice(shave_w if j else 0, (shave_w if j else 0) + split_w))
            plan.append((source, target, inner))
    return plan, (pad_h, pad_w)


def _scaled(slices, scale):
    return tuple(slice(part.start * scale, part.stop * scale) for part in slices)


class AuditHooks:
    """Temporary forward hooks observe the production computation unchanged."""
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.reset()
        for stage in model.layers[3:]:
            for layer in stage.residual_group.layers:
                self.handles.append(layer.assm.route.register_forward_hook(self.route))
        self.handles.append(model.udr.register_forward_pre_hook(self.before_udr))
        self.handles.append(model.udr.gre.register_forward_hook(self.confidence))
        self.handles.append(model.udr.projection.register_forward_hook(self.residual))
        self.handles.append(model.udr.register_forward_hook(self.after_udr))

    def reset(self):
        self.routes = []
        self.parts = {}
        self.shapes = {}
        self.maps = None

    def route(self, module, inputs, log_probability):
        p = log_probability.float().softmax(-1)
        top = p.topk(2, dim=-1).values
        entropy = -(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(p.shape[-1])
        self.routes.append((entropy, top[..., 0], top[..., 0] - top[..., 1]))
        self.shapes['pred_route'] = list(log_probability.shape)
        self.shapes['routing_probs'] = list(p.shape)

    def before_udr(self, module, inputs):
        feature, rgb, depth, ambiguity = inputs
        self.parts['feature'] = feature
        self.parts['depth'] = depth
        self.parts['ambiguity'] = ambiguity
        self.shapes['feature'] = list(feature.shape)
        self.shapes['ambiguity'] = list(ambiguity.shape)

    def confidence(self, module, inputs, output):
        self.parts['confidence'] = output
        self.shapes['confidence'] = list(output.shape)

    def residual(self, module, inputs, output):
        self.parts['residual'] = output
        self.shapes['depth_residual'] = list(output.shape)

    def after_udr(self, module, inputs, output):
        feature = self.parts['feature']
        ambiguity = self.parts['ambiguity']
        confidence = self.parts['confidence']
        residual = self.parts['residual']
        gate = ambiguity * confidence
        correction = output[0] - feature
        # E3 replaces only scalar alpha with a spatial map. Read the actual
        # forward value so the diagnostic assertion covers both E0 and E3.
        strength = (module.local_alpha_map if hasattr(module, 'local_alpha_map')
                    else module.alpha)
        expected = strength * gate * residual
        torch.testing.assert_close(correction, expected, rtol=2e-3, atol=1e-6)
        h, w = ambiguity.shape[-2:]
        if len(self.routes) != sum(len(stage.residual_group.layers)
                                   for stage in self.model.layers[3:]):
            raise RuntimeError('Incomplete routing capture; expected every ASSM in stages 4-6.')
        route_maps = []
        for index in range(3):
            channels = torch.stack([entry[index].reshape(ambiguity.shape[0], 1, h, w)
                                    for entry in self.routes]).mean(0)
            route_maps.append(channels)
        # E0 consumes entropy exactly; E2 deliberately supplies a different
        # uncertainty map to the same late UDR expert. Keep both observable.
        if not hasattr(self.model, 'uncertainty_mode'):
            torch.testing.assert_close(route_maps[0].clamp(0, 1), ambiguity.float(),
                                       rtol=1e-5, atol=1e-6)
        self.maps = dict(zip(MAP_KEYS, (
            route_maps[0], route_maps[1], route_maps[2], ambiguity,
            confidence, gate, module.gradient(self.parts['depth']),
            residual.abs().mean(1, keepdim=True),
            correction.abs().mean(1, keepdim=True),
            correction.square().mean(1, keepdim=True),
            correction.abs().amax(1, keepdim=True),
            feature.abs().mean(1, keepdim=True))))
        if hasattr(module, 'local_alpha_map'):
            self.maps['local_alpha'] = module.local_alpha_map
            self.shapes['local_alpha'] = list(module.local_alpha_map.shape)
        self.parts['correction'] = correction
        self.shapes['gate'] = list(gate.shape)
        self.shapes['correction'] = list(correction.shape)
        self.shapes['depth_gradient'] = list(self.maps['depth_gradient'].shape)

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def infer_partitioned(model, rgb, depth=None, audit=False):
    """Return a stitched HR prediction and, for UDR, LR audit maps."""
    if rgb.shape[0] != 1:
        raise ValueError('E0 expects batch size 1 for per-image correlations.')
    height, width = rgb.shape[-2:]
    plan, (pad_h, pad_w) = partition_plan(height, width)
    padded = F.pad(rgb, (0, pad_w, 0, pad_h), mode='reflect')
    if depth is not None:
        if depth.shape != rgb[:, :1].shape:
            raise ValueError('RGB/depth geometry mismatch.')
        depth = F.pad(depth, (0, pad_w, 0, pad_h), mode='reflect')
    scale = model.upscale
    prediction = torch.zeros(1, 3, padded.shape[-2] * scale,
                             padded.shape[-1] * scale, dtype=rgb.dtype)
    stitched = {}
    shapes = None
    hook = AuditHooks(model) if audit else None
    try:
        for source, target, inner in plan:
            if hook is not None:
                hook.reset()
            crop = padded[..., source[0], source[1]]
            depth_crop = None if depth is None else depth[..., source[0], source[1]]
            out = model(crop) if depth_crop is None else model(crop, depth_crop)
            dst = _scaled(target, scale)
            src = _scaled(inner, scale)
            prediction[..., dst[0], dst[1]] = out[..., src[0], src[1]].cpu()
            if hook is not None:
                if hook.maps is None:
                    raise RuntimeError('UDR hooks did not observe a complete forward pass.')
                if shapes is None:
                    shapes = hook.shapes.copy()
                for key, value in hook.maps.items():
                    if key not in stitched:
                        stitched[key] = torch.zeros(1, 1, padded.shape[-2],
                                                    padded.shape[-1], dtype=torch.float32)
                    stitched[key][..., target[0], target[1]] = (
                        value[..., inner[0], inner[1]].float().cpu())
        maps = {key: value[..., :height, :width].squeeze().numpy()
                for key, value in stitched.items()}
        return prediction[..., :height * scale, :width * scale], maps, shapes
    finally:
        if hook is not None:
            hook.close()


def correlation(a, b):
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Correlation maps must be aligned and finite.')
    # A constant map has undefined correlation. Preserve that as null in JSON/CSV.
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None, None
    return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])


def image_statistics(maps, rgb_error, alpha):
    if rgb_error.shape != maps['ambiguity'].shape:
        raise ValueError('HR error was not aligned with the LR audit maps.')
    out = {}
    for key in ('route_entropy', 'ambiguity', 'confidence', 'gate'):
        values = maps[key]
        out[key + '_mean'] = float(values.mean())
        out[key + '_std'] = float(values.std())
    for key in ('ambiguity',):
        out.update({key + '_p' + label: float(np.percentile(maps[key], percentile))
                    for label, percentile in [('05', 5), ('25', 25), ('50', 50),
                                              ('75', 75), ('95', 95)]})
    out['route_maxprob_mean'] = float(maps['route_maxprob'].mean())
    out['route_margin_mean'] = float(maps['route_margin'].mean())
    out['alpha'] = float(alpha)
    out['correction_rms'] = float(np.sqrt(maps['correction_square_mean'].mean()))
    out['correction_abs_mean'] = float(maps['correction_abs'].mean())
    out['correction_abs_max'] = float(maps['correction_channel_max'].max())
    out['gate_confidence_abs_difference'] = float(np.mean(np.abs(maps['gate'] - maps['confidence'])))
    out['gate_confidence_pearson'], _ = correlation(maps['gate'], maps['confidence'])
    out['rgb_error_mean'] = float(rgb_error.mean())
    for key in CORRELATION_KEYS:
        out[key + '_rgb_error_pearson'], out[key + '_rgb_error_spearman'] = (
            correlation(maps[key], rgb_error))
    correction = maps['correction_abs'].reshape(-1)
    error = rgb_error.reshape(-1)
    k = max(1, math.ceil(error.size * 0.1))
    top_error = np.argpartition(error, -k)[-k:]
    top_correction = np.argpartition(correction, -k)[-k:]
    out['correction_top10pct_error_overlap'] = float(
        np.intersect1d(top_error, top_correction).size / k)
    out['correction_high_error_mean'] = float(correction[top_error].mean())
    out['correction_other_error_mean'] = float(np.delete(correction, top_error).mean())
    return out


def aggregate(rows, fields):
    result = {}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None
                  and math.isfinite(float(row[field]))]
        result[field] = dict(mean=float(np.mean(values)) if values else None,
                             median=float(np.median(values)) if values else None,
                             std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                             valid=len(values), total=len(rows))
    return result


def _rgb(tensor):
    return tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)


def visualize(path, sample, rgb_sr, udr_sr, rgb_error_hr, maps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [
        ('LR RGB', _rgb(sample['lq']), None), ('GT', _rgb(sample['gt']), None),
        ('RGB-only SR', _rgb(rgb_sr), None), ('UDR SR', _rgb(udr_sr), None),
        ('RGB error |SR-GT|', rgb_error_hr, 'magma'),
        ('Depth', sample['depth'].squeeze().numpy(), 'viridis'),
        ('Depth gradient', maps['depth_gradient'], 'magma'),
        ('Routing ambiguity', maps['ambiguity'], 'viridis'),
        ('Depth confidence', maps['confidence'], 'viridis'),
        ('Gate', maps['gate'], 'viridis'),
        ('Correction |feature|', maps['correction_abs'], 'magma')]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    for ax, (title, image, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap, vmin=0 if cmap else None,
                  vmax=(np.percentile(image, 99) + 1e-12) if cmap == 'magma' else
                       (1 if cmap == 'viridis' and title not in ('Depth gradient',) else None))
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    axes.flat[-1].axis('off')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_config(path):
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def load_weights(model, path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location='cpu')
    if not isinstance(state, dict) or 'params' not in state:
        raise ValueError('Checkpoint must contain a params state dict: ' + str(path))
    state = {(key[7:] if key.startswith('module.') else key): value
             for key, value in state['params'].items()}
    model.load_state_dict(state, strict=True)


def run(args):
    from basicsr.archs.mambairv2_arch import MambaIRv2
    from basicsr.archs.udr_mambairv2_arch import UDRMambaIRv2
    from basicsr.data import build_dataset
    from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
    from basicsr.utils.img_util import tensor2img

    if not torch.cuda.is_available() and args.device == 'cuda':
        raise RuntimeError('The production E0 audit requires installed Mamba CUDA support.')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use distinct matched seeds.')
    phase_a = load_config(args.train_config)
    reference = load_config(args.rgb_config)
    udr_test = load_config(args.udr_config)
    if reference['network_g']['type'] != 'MambaIRv2' or udr_test['network_g']['type'] != 'UDRMambaIRv2':
        raise ValueError('Expected fixed RGB baseline and v3.2 UDR architecture configs.')
    rgb_options = dict(reference['network_g'])
    rgb_options.pop('type')
    udr_options = dict(udr_test['network_g'])
    udr_options.pop('type')
    baseline = MambaIRv2(**rgb_options)
    udr = UDRMambaIRv2(**udr_options)
    load_weights(baseline, args.rgb_checkpoint or reference['path']['pretrain_network_g'])
    load_weights(udr, args.udr_checkpoint or udr_test['path']['pretrain_network_g'])
    baseline = baseline.to(args.device).eval()
    udr = udr.to(args.device).eval()
    data_options = dict(phase_a['datasets']['val'])
    data_options.update(phase='val', scale=4)
    dataset = build_dataset(data_options)
    if len(dataset) < args.visualize_count:
        raise ValueError('DIV2K validation has fewer images than the requested visualization count.')
    limit = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    if limit < args.visualize_count:
        raise ValueError('max-images cannot be smaller than visualize-count.')
    output = Path(args.output).resolve()
    pictures = output/'E0_visualization'
    pictures.mkdir(parents=True, exist_ok=True)
    rows, shape_record = [], None
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for index in range(limit):
        sample = dataset[index]
        image_id = Path(sample['lq_path']).stem
        rgb = sample['lq'].unsqueeze(0)
        depth = sample['depth'].unsqueeze(0)
        gt = sample['gt'].unsqueeze(0)
        for seed in args.seeds:
            # Both networks see the same Gumbel random stream from image start.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            rgb_sr, _, _ = infer_partitioned(baseline, rgb.to(args.device))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            udr_sr, maps, shapes = infer_partitioned(
                udr, rgb.to(args.device), depth.to(args.device), audit=True)
            if shape_record is None:
                shape_record = shapes
            if rgb_sr.shape != gt.shape or udr_sr.shape != gt.shape:
                raise ValueError('SR output and GT shapes differ; no implicit resizing.')
            # Compare all LR diagnostic maps with baseline RGB channel-mean L1
            # error, downsampled by aligned 4x4 area averages. This is a proxy,
            # separate from the Y-channel/crop-border PSNR evaluation.
            error_hr = (rgb_sr - gt).abs().mean(1, keepdim=True)
            error_lr = F.avg_pool2d(error_hr, kernel_size=4, stride=4).squeeze().numpy()
            metrics = image_statistics(maps, error_lr, udr.udr.alpha.detach().cpu().item())
            gt_img = tensor2img(gt)
            rgb_img, udr_img = tensor2img(rgb_sr), tensor2img(udr_sr)
            for label, image in (('rgb', rgb_img), ('udr', udr_img)):
                metrics[label + '_psnr'] = float(calculate_psnr(
                    image, gt_img, crop_border=4, test_y_channel=True))
                metrics[label + '_ssim'] = float(calculate_ssim(
                    image, gt_img, crop_border=4, test_y_channel=True))
            row = dict(image_id=image_id, seed=seed,
                       delta_psnr=metrics['udr_psnr']-metrics['rgb_psnr'],
                       delta_ssim=metrics['udr_ssim']-metrics['rgb_ssim'], **metrics)
            rows.append(row)
            if seed == args.seeds[0] and index < args.visualize_count:
                visualize(pictures/(f'{index + 1:03d}_{image_id}.png'), sample,
                          rgb_sr, udr_sr, error_hr.squeeze().numpy(), maps)
        print(f'E0 {index+1}/{limit}: {image_id}', flush=True)

    csv_path = output/'E0_mechanism_audit.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fields = LOG_KEYS + tuple(key + '_rgb_error_' + metric for key in CORRELATION_KEYS
                              for metric in ('pearson', 'spearman')) + (
        'gate_confidence_abs_difference', 'gate_confidence_pearson',
        'correction_top10pct_error_overlap', 'correction_high_error_mean',
        'correction_other_error_mean', 'delta_psnr', 'delta_ssim')
    aggregated = aggregate(rows, fields)
    u_rho = aggregated['ambiguity_rgb_error_spearman']['mean']
    u_std = aggregated['ambiguity_std']['mean']
    u_range = aggregate(rows, ('ambiguity_p95', 'ambiguity_p05'))
    # These cutoffs are an explicit audit convention, not a learned calibration.
    dynamic = u_range['ambiguity_p95']['mean'] - u_range['ambiguity_p05']['mean']
    if u_rho is not None and u_rho > .2 and dynamic >= .05:
        decision = 'retain routing ambiguity proxy for follow-up testing'
    elif u_rho is not None and .05 < u_rho <= .2:
        decision = 'calibrate or replace; positive but weak error association'
    elif u_rho is not None and abs(u_rho) <= .05 and u_std is not None and u_std < .01:
        decision = 'abandon routing entropy as RGB reconstruction uncertainty proxy'
    else:
        decision = 'inconclusive; inspect maps and per-image correlations'
    payload = dict(experiment='E0 UDR mechanism audit', model_training='none',
                   dataset=data_options['name'], image_count=limit, seeds=args.seeds,
                   row_count=len(rows), visualized_images=args.visualize_count,
                   checkpoint_paths=dict(rgb=args.rgb_checkpoint or reference['path']['pretrain_network_g'],
                                         udr=args.udr_checkpoint or udr_test['path']['pretrain_network_g']),
                   tensor_shapes_first_partition=shape_record,
                   rgb_error_definition='mean RGB absolute SR-GT error, area-averaged x4 to LR',
                   performance_metric='original uint8 Y-channel PSNR/SSIM, crop border 4',
                   partition_protocol='MambaIRv2Model.test borders and overlap',
                   statistics=aggregated, routing_ambiguity_decision=decision,
                   decision_cutoffs=dict(retain_spearman=.2, minimum_p95_minus_p05=.05,
                                         weak_spearman=.05, near_constant_std=.01),
                   note='Correlations are descriptive. Alpha shrinkage and negative transfer are hypotheses, not proven causes.')
    (output/'E0_statistics.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# E0 UDR mechanism audit', '',
             f'DIV2K validation: {limit} images × {len(args.seeds)} matched seeds; no training.',
             'RGB error = channel-mean |RGB-only SR − GT|, area-averaged to LR.',
             'PSNR/SSIM use original Y-channel, crop-border 4 protocol.', '',
             '| Quantity | Mean across image × seed | Valid rows |', '|---|---:|---:|']
    for key in fields:
        item = aggregated[key]
        value = 'undefined' if item['mean'] is None else f"{item['mean']:.6g}"
        lines.append(f"| {key} | {value} | {item['valid']}/{item['total']} |")
    lines += ['', f'Routing ambiguity P95−P05 (difference of aggregate means): {dynamic:.6g}.',
              f'Working decision: **{decision}**.', '',
              'If entropy is almost uniform, gate ≈ confidence by construction. '
              'A smaller trained alpha than its 0.01 initialization is consistent with '
              'suppression but does not establish why optimization chose it.',
              'Inspect correction/high-error overlap and maps before causal claims.',
              '', 'Files: E0_mechanism_audit.csv, E0_statistics.json, '
              'E0_visualization/, E0_summary.md.', '']
    (output/'E0_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:6]))
    print('Saved E0 audit to', output)


def self_test():
    class Toy(torch.nn.Module):
        upscale = 4
        def forward(self, x, depth=None):
            if depth is not None:
                torch.testing.assert_close(x[:, :1], depth)
            return F.interpolate(x, scale_factor=4, mode='nearest')
    rgb = torch.rand(1, 3, 203, 407)
    result, maps, shapes = infer_partitioned(Toy(), rgb)
    torch.testing.assert_close(result, F.interpolate(rgb, scale_factor=4, mode='nearest'))
    assert not maps and shapes is None
    assert correlation(np.zeros((4, 4)), np.ones((4, 4))) == (None, None)
    line = np.arange(16).reshape(4, 4).astype(float)
    assert correlation(line, line)[1] == 1
    fake = {key: line / 15 for key in MAP_KEYS}
    fake['correction_abs'] = line / 150
    row = image_statistics(fake, line / 15, .003)
    assert row['ambiguity_rgb_error_spearman'] == 1
    print('PASS: exact odd-size partition stitch, constant-map handling, correlations and E0 diagnostics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-config', default=str(ROOT/'options/train/mambairv2/train_UDR_MambaSR_x4_phaseA.yml'))
    parser.add_argument('--rgb-config', default=str(ROOT/'options/test/mambairv2/test_UDR_RGB_reference_x4.yml'))
    parser.add_argument('--udr-config', default=str(ROOT/'options/test/mambairv2/test_UDR_MambaSR_x4.yml'))
    parser.add_argument('--rgb-checkpoint')
    parser.add_argument('--udr-checkpoint')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=[10, 11, 12])
    parser.add_argument('--max-images', type=int, default=0,
                        help='0 means all DIV2K validation images.')
    parser.add_argument('--visualize-count', type=int, default=20)
    parser.add_argument('--output', default=str(ROOT/'results/E0_udr_mechanism_audit'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == '__main__':
    main()
