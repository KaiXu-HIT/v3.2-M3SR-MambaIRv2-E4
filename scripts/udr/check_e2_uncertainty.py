"""Offline E2 contract checks using the existing differentiable CPU scan reference."""
import ast
from copy import deepcopy
import logging
from pathlib import Path
import sys
import tempfile
import types

import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.grs.check_grs import load_isolated
from scripts.udr.check_udr import load_model_class
from scripts.udr.e0_mechanism_audit import infer_partitioned


def configs():
    source = yaml.safe_load((ROOT/'options/train/mambairv2/train_UDR_MambaSR_x4_phaseB.yml').read_text())
    rgb = yaml.safe_load((ROOT/'options/test/mambairv2/test_UDR_RGB_reference_x4.yml').read_text())
    original_test = yaml.safe_load((ROOT/'options/test/mambairv2/test_UDR_MambaSR_x4.yml').read_text())
    for label in ('U1', 'U2', 'U3'):
        phases = ('A', 'B') if label == 'U3' else ('B',)
        for phase in phases:
            item = yaml.safe_load((ROOT/f'options/train/mambairv2/train_E2_{label}_phase{phase}_x4.yml').read_text())
            assert item['datasets'] == source['datasets']
            assert item['train']['pixel_opt'] == source['train']['pixel_opt']
            assert item['train']['phase'] == phase
            assert item['train']['total_iter'] == (30000 if phase == 'A' else 100000)
            assert item['train']['lambda_u'] == (.01 if label == 'U3' else 0)
            assert item['train']['rgb_teacher_checkpoint'] == rgb['path']['pretrain_network_g']
        test = yaml.safe_load((ROOT/f'options/test/mambairv2/test_E2_{label}_x4.yml').read_text())
        assert test['datasets'] == original_test['datasets']
        assert test['val'] == original_test['val']
        assert test['scale'] == 4
    print('PASS: all seven E2 configs preserve original data/metrics and phase budgets')


def check_architecture():
    base = load_isolated((ROOT/'basicsr/archs/mambairv2_arch.py').read_text())
    udr = load_isolated((ROOT/'basicsr/archs/udr_mambairv2_arch.py').read_text(),
                        dict(MambaIRv2=base.MambaIRv2))
    e2 = load_isolated((ROOT/'basicsr/archs/e2_udr_mambairv2_arch.py').read_text(),
                       dict(UDRMambaIRv2=udr.UDRMambaIRv2))
    options = dict(img_size=8, embed_dim=12, depths=(1,) * 6,
                   num_heads=(3,) * 6, window_size=4, d_state=2,
                   inner_rank=4, num_tokens=4, mlp_ratio=1.,
                   upscale=4, upsampler='pixelshuffle')
    torch.manual_seed(5)
    source = udr.UDRMambaIRv2(**options).eval()
    old_state = source.state_dict()
    rgb, depth = torch.rand(1, 3, 8, 8), torch.rand(1, 1, 8, 8)
    for mode in e2.E2UDRMambaIRv2.MODES:
        model = e2.E2UDRMambaIRv2(uncertainty_mode=mode, **options).eval()
        model.load_e0_state_dict(old_state)
        assert all(torch.equal(model.state_dict()[k], old_state[k]) for k in old_state)
        torch.manual_seed(11)
        out = model(rgb, depth)
        assert out.shape == (1, 3, 32, 32)
        assert model.uncertainty_map.shape == (1, 1, 8, 8)
        assert torch.all((model.uncertainty_map >= 0) & (model.uncertainty_map <= 1))
        assert torch.isfinite(model.uncertainty_map).all()
        assert model.udr_stats['gate_std'].isfinite()
        # E0's partition hook must observe the actual E2 gate input, while
        # retaining route entropy as a separate diagnostic map.
        _, maps, _ = infer_partitioned(model, rgb, depth, audit=True)
        torch.testing.assert_close(torch.from_numpy(maps['ambiguity']),
                                   model.uncertainty_map.squeeze().detach(),
                                   atol=1e-5, rtol=1e-5)
        if mode == 'route_concentration':
            assert model.uncertainty_map.max() <= .75 + 1e-5  # 1 - 1/K
        if mode == 'learned_error':
            model.configure_phase('A')
            assert all(p.requires_grad == n.startswith('uncertainty_head.')
                       for n, p in model.named_parameters())
            model.train()
            assert not model.udr.training and not model.layers[0].training
            assert model.uncertainty_head.training
            torch.manual_seed(12)
            value = model(rgb, depth)
            target = torch.rand_like(model.uncertainty_map)
            (value.abs().mean() + .01 * F.l1_loss(model.uncertainty_map, target)).backward()
            assert model.uncertainty_head[0].weight.grad is not None
            assert model.udr.projection.weight.grad is None
            model.configure_phase('B')
            assert all(p.requires_grad for p in model.parameters())
    print('PASS: strict E0 transfer, U1/U2/U3 maps, U3 A/B freeze and head gradient')
    return e2


def check_error_target():
    source = (ROOT/'basicsr/models/e2_udr_mambairv2_model.py').read_text()
    # Exercise the production target function independently of BasicSR/CUDA.
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'normalized_rgb_error')
    scope = dict(torch=torch, F=F)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<target>', 'exec'), scope)
    target = scope['normalized_rgb_error']
    gt = torch.zeros(2, 3, 16, 16)
    sr = torch.ones_like(gt)
    t = target(sr, gt)
    assert t.shape == (2, 1, 4, 4)
    torch.testing.assert_close(t, torch.ones_like(t))
    torch.testing.assert_close(target(gt, gt), torch.zeros_like(t))
    print('PASS: fixed RGB reconstruction error normalizes and aligns to LR')


def check_training():
    base = load_isolated((ROOT/'basicsr/archs/mambairv2_arch.py').read_text())
    udr = load_isolated((ROOT/'basicsr/archs/udr_mambairv2_arch.py').read_text(),
                        dict(MambaIRv2=base.MambaIRv2))
    e2 = load_isolated((ROOT/'basicsr/archs/e2_udr_mambairv2_arch.py').read_text(),
                       dict(UDRMambaIRv2=udr.UDRMambaIRv2))
    # Reuse the real BasicSR model source with only the unavailable CUDA scan
    # swapped for its differentiable CPU reference.
    E0Model = load_model_class(types.SimpleNamespace(UDRMambaIRv2=e2.E2UDRMambaIRv2), True)
    class Registry:
        def register(self):
            return lambda cls: cls
    e2_model = load_isolated((ROOT/'basicsr/models/e2_udr_mambairv2_model.py').read_text(),
                             dict(UDRMambaIRv2Model=E0Model, MambaIRv2=base.MambaIRv2,
                                  SRModel=E0Model.__mro__[1], MODEL_REGISTRY=Registry(),
                                  deepcopy=deepcopy,
                                  get_root_logger=lambda: logging.getLogger('e2_check')))
    kw = dict(img_size=8, embed_dim=12, depths=(1,) * 6,
              num_heads=(3,) * 6, window_size=4, d_state=2,
              inner_rank=4, num_tokens=4, mlp_ratio=1.,
              upscale=4, upsampler='pixelshuffle')
    old = udr.UDRMambaIRv2(**kw)
    teacher = base.MambaIRv2(**kw)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        old_path, teacher_path, phase_a_path = (folder/name for name in
                                                 ('e0.pth', 'rgb.pth', 'e2_a.pth'))
        torch.save(dict(params=old.state_dict()), old_path)
        torch.save(dict(params=teacher.state_dict()), teacher_path)
        options = dict(is_train=True, dist=False, num_gpu=0,
                       network_g=dict(type='E2UDRMambaIRv2', uncertainty_mode='learned_error', **kw),
                       path=dict(pretrain_network_g=str(old_path), pretrain_kind='e0',
                                 strict_load_g=True),
                       train=dict(phase='A', lambda_u=.01,
                                  rgb_teacher_checkpoint=str(teacher_path),
                                  mechanism_log_interval=500,
                                  optim_g=dict(type='Adam', lr=1e-4),
                                  scheduler=dict(type='MultiStepLR', milestones=[], gamma=1.),
                                  pixel_opt=dict(type='L1Loss', loss_weight=1., reduction='mean')))
        sample = dict(lq=torch.rand(1, 3, 4, 4), depth=torch.rand(1, 1, 4, 4),
                      gt=torch.rand(1, 3, 16, 16))
        a = e2_model.E2UDRMambaIRv2Model(deepcopy(options))
        before = {n: p.detach().clone() for n, p in a.get_bare_model(a.net_g).named_parameters()}
        a.feed_data(sample)
        a.optimize_parameters(1)
        net = a.get_bare_model(a.net_g)
        for name, parameter in net.named_parameters():
            if name.startswith('uncertainty_head.'):
                assert parameter.grad is not None, name
            else:
                assert parameter.grad is None, name
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        assert 'l_u' in a.log_dict and a.optimizer_g.param_groups[0]['group_name'] == 'uncertainty'
        torch.save(dict(params=net.state_dict()), phase_a_path)
        bcfg = deepcopy(options)
        bcfg['train'].update(phase='B', rgb_lr=1e-5)
        bcfg['path'].update(pretrain_network_g=str(phase_a_path), pretrain_kind='e2')
        b = e2_model.E2UDRMambaIRv2Model(bcfg)
        b.feed_data(sample)
        b.optimize_parameters(1)
        bnet = b.get_bare_model(b.net_g)
        assert bnet.udr.projection.weight.grad is not None
        assert bnet.conv_first.weight.grad is not None
        assert bnet.uncertainty_head[0].weight.grad is not None
        assert [g['lr'] for g in b.optimizer_g.param_groups] == [1e-4, 1e-4, 1e-5]
        for mode in ('route_concentration', 'feature_variance'):
            probe = deepcopy(options)
            probe['network_g']['uncertainty_mode'] = mode
            probe['train'].update(phase='B', lambda_u=0, rgb_lr=1e-5)
            u = e2_model.E2UDRMambaIRv2Model(probe)
            u.feed_data(sample)
            u.optimize_parameters(1)
            assert 'l_u' not in u.log_dict
            assert [g['lr'] for g in u.optimizer_g.param_groups] == [1e-4, 1e-5]
    print('PASS: E2 U1/U2 joint steps and U3 A/B steps, frozen teacher, strict transfer')


def main():
    torch.set_num_threads(2)
    configs()
    check_architecture()
    check_error_target()
    check_training()
    print('ALL E2 CPU REFERENCE CHECKS PASSED')


if __name__ == '__main__':
    main()
