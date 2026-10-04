"""Select measured E2/E3 winners and prepare a strict E4 initialization.

Requires the actual upstream reports/checkpoints. It never guesses a winner or
fills missing scores. Shared RGB/Depth tensors come from the selected E3 model;
only a learned U3 uncertainty head is transplanted from selected E2. U1/U2
have no learned uncertainty weights. Generated YAML preserves E3 data paths.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODES = dict(U1='route_concentration', U2='feature_variance',
             U3='learned_error')
TAGS = dict(a0=0., a1e5=1e-5, a1e4=1e-4, a5e4=5e-4)


def read_json(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'E4 needs measured upstream report: {path}')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dataset_delta(report, name):
    return float(report['summary'][name]['udr']['delta_psnr_vs_rgb'])


def choose_variants(e2_root, e3_root, e0_statistics):
    """Select measured mechanism-valid U and spatial A; then maximize 5-set gain."""
    old = read_json(e0_statistics)
    e0_rho = old['statistics']['ambiguity_rgb_error_spearman']['mean']
    if e0_rho is None:
        raise ValueError('E0 uncertainty correlation is undefined; cannot rank E2.')
    u_candidates = []
    for variant, mode in MODES.items():
        perf = read_json(Path(e2_root)/f'{variant}_five_set/summary.json')
        audit = read_json(Path(e2_root)/f'{variant}_audit/E2_statistics.json')
        if audit['variant'] != mode:
            raise ValueError(f'E2 {variant} audit mode mismatches checkpoint/config.')
        rho = audit['statistics']['uncertainty_rgb_error_spearman']['mean']
        if rho is None or not math.isfinite(float(rho)):
            raise ValueError(f'E2 {variant} uncertainty correlation undefined.')
        score = float(perf['average_delta_vs_rgb']['udr'])
        u_candidates.append(dict(variant=variant, mode=mode,
                                 average_delta_psnr=score,
                                 urban_delta=dataset_delta(perf, 'Urban100'),
                                 manga_delta=dataset_delta(perf, 'Manga109'),
                                 spearman=float(rho),
                                 gain_over_e0=float(rho) - float(e0_rho),
                                 mechanism_go=float(rho) > .2 and
                                 float(rho) - float(e0_rho) >= .05))
    eligible_u = [item for item in u_candidates if item['mechanism_go']]
    if not eligible_u:
        raise ValueError('No E2 variant meets the documented Go criterion; '
                         'E4 cannot honestly call any U_best. Inspect E2 reports.')
    best_u = max(eligible_u, key=lambda item: (item['average_delta_psnr'],
                                                item['urban_delta'],
                                                item['manga_delta']))
    a_candidates = []
    for tag, lam in TAGS.items():
        perf = read_json(Path(e3_root)/f'{tag}_five_set/summary.json')
        audit = read_json(Path(e3_root)/f'{tag}_audit/E3_statistics.json')
        span = audit['mean_p95_minus_p05']
        std = audit['statistics']['local_alpha_std']['mean']
        if span is None or std is None or not math.isfinite(float(span)) or not math.isfinite(float(std)):
            raise ValueError(f'E3 {tag} local-alpha variation is undefined.')
        a_candidates.append(dict(tag=tag, lambda_a=lam,
                                 average_delta_psnr=float(perf['average_delta_vs_rgb']['udr']),
                                 urban_delta=dataset_delta(perf, 'Urban100'),
                                 manga_delta=dataset_delta(perf, 'Manga109'),
                                 local_alpha_std=float(std),
                                 local_alpha_p95_minus_p05=float(span),
                                 spatial=float(std) > 0 and float(span) > 0))
    eligible_a = [item for item in a_candidates if item['spatial']]
    if not eligible_a:
        raise ValueError('No E3 variant has a spatially varying A_D map.')
    best_a = max(eligible_a, key=lambda item: (item['average_delta_psnr'],
                                                item['urban_delta'],
                                                item['manga_delta']))
    return dict(e0_uncertainty_spearman=float(e0_rho),
                e2_candidates=u_candidates, e3_candidates=a_candidates,
                selected_uncertainty=best_u, selected_alpha=best_a,
                selection_rule='E2: Spearman>0.2 and >=+0.05 versus E0, then max five-set average delta; E3: spatial map, then max five-set average delta; Urban/Manga tie breaks')


def parameters(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Missing selected checkpoint: {path}')
    state = torch.load(path, map_location='cpu')
    if not isinstance(state, dict) or not isinstance(state.get('params'), dict):
        raise ValueError(f'Checkpoint has no params state dict: {path}')
    return {(key[7:] if key.startswith('module.') else key): value
            for key, value in state['params'].items()}


def merge_parameters(e2_state, e3_state, variant):
    """Use E3's shared tensors; inject only E2-U3's trained head, if present."""
    local = {key for key in e3_state if key.startswith('udr.local_alpha_head.')}
    uncertain = {key for key in e2_state if key.startswith('uncertainty_head.')}
    if not local:
        raise ValueError('Selected E3 checkpoint has no trained local-alpha head.')
    if (variant == 'U3') != bool(uncertain):
        raise ValueError('Selected E2 checkpoint does not match its U1/U2/U3 mode.')
    if any(key.startswith('uncertainty_head.') for key in e3_state):
        raise ValueError('E3 checkpoint unexpectedly contains E2 head weights.')
    if any(key.startswith('udr.local_alpha_head.') for key in e2_state):
        raise ValueError('E2 checkpoint unexpectedly contains E3 head weights.')
    shared_e3 = set(e3_state) - local
    shared_e2 = set(e2_state) - uncertain
    if shared_e2 != shared_e3:
        raise ValueError(f'E2/E3 shared tensor schemas differ: '
                         f'E2-only={sorted(shared_e2-shared_e3)}, '
                         f'E3-only={sorted(shared_e3-shared_e2)}')
    for key in shared_e3:
        if e2_state[key].shape != e3_state[key].shape:
            raise ValueError(f'Incompatible E2/E3 shared tensor shape: {key}')
    merged = dict(e3_state)
    merged.update({key: e2_state[key] for key in uncertain})
    return merged


def generate_configs(tag, mode, initial_checkpoint, rgb_teacher_checkpoint):
    """Write fixed-name A/B/test YAML so commands never imply a guessed winner."""
    train_dir = ROOT/'options/train/mambairv2'
    test_dir = ROOT/'options/test/mambairv2'
    outputs = {}
    for phase in ('A', 'B'):
        source = train_dir/f'train_E3_{tag}_phase{phase}_x4.yml'
        config = yaml.safe_load(source.read_text(encoding='utf-8'))
        config['name'] = f'E4_selected_phase{phase}_x4'
        config['model_type'] = 'E4UDRMambaIRv2Model'
        config['network_g']['type'] = 'E4UDRMambaIRv2'
        config['network_g']['uncertainty_mode'] = mode
        config['train']['lambda_u'] = .01 if mode == 'learned_error' else 0.
        if mode == 'learned_error':
            config['train']['rgb_teacher_checkpoint'] = rgb_teacher_checkpoint
        config['path']['pretrain_kind'] = 'e4'
        config['path']['pretrain_network_g'] = (
            str(initial_checkpoint) if phase == 'A' else
            'experiments/E4_selected_phaseA_x4/models/net_g_30000.pth')
        path = train_dir/f'train_E4_phase{phase}_x4.yml'
        path.write_text('# Generated from measured E2/E3 winners; do not change dataset paths.\n'
                        + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                        encoding='utf-8')
        outputs[f'train_{phase}'] = str(path)
    config = yaml.safe_load((test_dir/f'test_E3_{tag}_x4.yml').read_text(encoding='utf-8'))
    config['name'] = 'test_E4_selected_x4'
    config['model_type'] = 'E4UDRMambaIRv2Model'
    config['network_g']['type'] = 'E4UDRMambaIRv2'
    config['network_g']['uncertainty_mode'] = mode
    config['path']['pretrain_kind'] = 'e4'
    config['path']['pretrain_network_g'] = 'experiments/E4_selected_phaseB_x4/models/net_g_100000.pth'
    path = test_dir/'test_E4_x4.yml'
    path.write_text('# Generated from measured E2/E3 winners; original five-set protocol.\n'
                    + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                    encoding='utf-8')
    outputs['test'] = str(path)
    return outputs


def run(args):
    selection = choose_variants(args.e2_results, args.e3_results,
                                args.e0_statistics)
    u, a = selection['selected_uncertainty'], selection['selected_alpha']
    # Results normally live at <upstream repo>/results/<experiment>. Derive
    # the corresponding checkpoint root, so separate E2/E3 checkouts work.
    e2_repo = Path(args.e2_results).resolve().parents[1]
    e3_repo = Path(args.e3_results).resolve().parents[1]
    e2_path = Path(args.e2_checkpoint or
                   e2_repo/f"experiments/E2_{u['variant']}_joint_x4/models/net_g_100000.pth")
    e3_path = Path(args.e3_checkpoint or
                   e3_repo/f"experiments/E3_{a['tag']}_phaseB_x4/models/net_g_100000.pth")
    e2_state, e3_state = parameters(e2_path), parameters(e3_path)
    merged = merge_parameters(e2_state, e3_state, u['variant'])

    # Build the production architecture and require exact tensor compatibility
    # before writing any E4 checkpoint/configuration.
    from basicsr.archs.e4_udrv2_arch import E4UDRMambaIRv2
    e3_config = yaml.safe_load((ROOT/f"options/test/mambairv2/test_E3_{a['tag']}_x4.yml").read_text(encoding='utf-8'))
    network = deepcopy(e3_config['network_g'])
    network.pop('type')
    network['uncertainty_mode'] = u['mode']
    model = E4UDRMambaIRv2(**network)
    model.load_state_dict(merged, strict=True)
    del model

    rgb_path = args.rgb_teacher_checkpoint or yaml.safe_load(
        (ROOT/'options/test/mambairv2/test_UDR_RGB_reference_x4.yml').read_text(encoding='utf-8'))['path']['pretrain_network_g']
    if u['variant'] == 'U3' and not Path(rgb_path).is_file():
        raise FileNotFoundError(f'Fixed RGB teacher missing: {rgb_path}')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial = output/'E4_initial.pth'
    selection.update(e2_checkpoint=str(e2_path), e3_checkpoint=str(e3_path),
                     rgb_teacher_checkpoint=str(rgb_path),
                     initial_checkpoint=str(initial),
                     combination='E3 shared RGB/Depth/local-alpha tensors + E2 U3 head only (U1/U2 parameter-free)')
    torch.save(dict(params=merged, e4_selection=selection), initial)
    selection['generated_configs'] = generate_configs(
        a['tag'], u['mode'], initial, rgb_path)
    (output/'E4_selection.json').write_text(json.dumps(selection, indent=2,
                                                       ensure_ascii=False, allow_nan=False),
                                              encoding='utf-8')
    print(f"Selected E2 {u['variant']} and E3 {a['tag']}; "
          f"saved strict merged weights to {initial}")


def self_test():
    shared = dict(weight=torch.ones(2, 2), **{'udr.alpha_raw': torch.ones(1)})
    e2 = dict(shared, **{'uncertainty_head.0.weight': torch.ones(1) * 3})
    e3 = dict(shared, **{'udr.local_alpha_head.rgb_compress.weight': torch.ones(1) * 5})
    merged = merge_parameters(e2, e3, 'U3')
    assert merged['uncertainty_head.0.weight'].item() == 3
    assert merged['udr.local_alpha_head.rgb_compress.weight'].item() == 5
    without_head = merge_parameters(shared, e3, 'U1')
    assert set(without_head) == set(e3)
    assert torch.equal(without_head['weight'], e3['weight'])
    try:
        merge_parameters(e2, e3, 'U1')
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched U1/U3 source was accepted.')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        e2_root, e3_root = root/'e2', root/'e3'
        e0_path = root/'e0.json'
        e0_path.write_text(json.dumps(dict(statistics=dict(
            ambiguity_rgb_error_spearman=dict(mean=.1)))), encoding='utf-8')
        def report(delta):
            return dict(average_delta_vs_rgb=dict(udr=delta),
                        summary={name: dict(udr=dict(delta_psnr_vs_rgb=delta))
                                 for name in ('Urban100', 'Manga109')})
        for variant, mode in MODES.items():
            (e2_root/f'{variant}_five_set').mkdir(parents=True)
            (e2_root/f'{variant}_audit').mkdir(parents=True)
            gain, rho = {'U1': (.02, .12), 'U2': (.05, .24),
                         'U3': (.07, .31)}[variant]
            (e2_root/f'{variant}_five_set/summary.json').write_text(
                json.dumps(report(gain)), encoding='utf-8')
            (e2_root/f'{variant}_audit/E2_statistics.json').write_text(
                json.dumps(dict(variant=mode, statistics=dict(
                    uncertainty_rgb_error_spearman=dict(mean=rho)))), encoding='utf-8')
        for tag in TAGS:
            (e3_root/f'{tag}_five_set').mkdir(parents=True)
            (e3_root/f'{tag}_audit').mkdir(parents=True)
            gain = {'a0': .02, 'a1e5': .04, 'a1e4': .06, 'a5e4': .03}[tag]
            (e3_root/f'{tag}_five_set/summary.json').write_text(
                json.dumps(report(gain)), encoding='utf-8')
            (e3_root/f'{tag}_audit/E3_statistics.json').write_text(
                json.dumps(dict(mean_p95_minus_p05=.006, statistics=dict(
                    local_alpha_std=dict(mean=.003)))), encoding='utf-8')
        selected = choose_variants(e2_root, e3_root, e0_path)
        assert selected['selected_uncertainty']['variant'] == 'U3'
        assert selected['selected_alpha']['tag'] == 'a1e4'
        # Exercise the final YAML writer in isolation, including preservation
        # of every E3 dataset path and U3's fixed-teacher requirement.
        global ROOT
        original_root = ROOT
        ROOT = root/'configs'
        try:
            for subdir, names in (
                    ('options/train/mambairv2',
                     ('train_E3_a1e4_phaseA_x4.yml',
                      'train_E3_a1e4_phaseB_x4.yml')),
                    ('options/test/mambairv2', ('test_E3_a1e4_x4.yml',))):
                target = ROOT/subdir
                target.mkdir(parents=True)
                for name in names:
                    shutil.copy2(original_root/subdir/name, target/name)
            outputs = generate_configs('a1e4', 'learned_error',
                                       root/'E4_initial.pth', str(root/'rgb.pth'))
            source = yaml.safe_load((ROOT/'options/train/mambairv2/'
                                     'train_E3_a1e4_phaseA_x4.yml').read_text(encoding='utf-8'))
            for phase in ('A', 'B'):
                generated = yaml.safe_load(Path(outputs[f'train_{phase}']).read_text(encoding='utf-8'))
                assert generated['datasets'] == source['datasets']
                assert generated['model_type'] == 'E4UDRMambaIRv2Model'
                assert generated['network_g']['uncertainty_mode'] == 'learned_error'
                assert generated['train']['lambda_u'] == .01
                assert generated['train']['rgb_teacher_checkpoint'] == str(root/'rgb.pth')
            test = yaml.safe_load(Path(outputs['test']).read_text(encoding='utf-8'))
            assert test['network_g']['type'] == 'E4UDRMambaIRv2'
        finally:
            ROOT = original_root
    print('PASS: measured winner selection, strict source-schema checks, U3 head transplant and generated YAML')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e2-results', default=str(ROOT/'results/E2_udrv2_uncertainty'))
    parser.add_argument('--e3-results', default=str(ROOT/'results/E3_local_alpha'))
    parser.add_argument('--e0-statistics', default=str(ROOT/'results/E0_udr_mechanism_audit/E0_statistics.json'))
    parser.add_argument('--e2-checkpoint', help='Selected E2 full checkpoint, if stored outside this checkout.')
    parser.add_argument('--e3-checkpoint', help='Selected E3 full checkpoint, if stored outside this checkout.')
    parser.add_argument('--rgb-teacher-checkpoint', help='Fixed RGB-only checkpoint, required for selected U3.')
    parser.add_argument('--output', default=str(ROOT/'experiments/E4_selection'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else run(args)


if __name__ == '__main__':
    main()
