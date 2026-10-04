"""Offline E3 contracts with the project's differentiable CPU scan reference."""
from copy import deepcopy
import logging
from pathlib import Path
import sys
import tempfile
import types

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.grs.check_grs import load_isolated
from scripts.udr.check_udr import load_model_class
from scripts.udr.e0_mechanism_audit import infer_partitioned

TAGS = dict(a0=0., a1e5=1e-5, a1e4=1e-4, a5e4=5e-4)


def check_configs():
    source = yaml.safe_load((ROOT/'options/train/mambairv2/train_UDR_MambaSR_x4_phaseB.yml').read_text())
    test_source = yaml.safe_load((ROOT/'options/test/mambairv2/test_UDR_MambaSR_x4.yml').read_text())
    for tag, strength in TAGS.items():
        for phase in ('A', 'B'):
            item = yaml.safe_load((ROOT/f'options/train/mambairv2/train_E3_{tag}_phase{phase}_x4.yml').read_text())
            assert item['datasets'] == source['datasets']
            assert item['val'] == source['val']
            assert item['train']['pixel_opt'] == source['train']['pixel_opt']
            assert item['train']['total_iter'] == (30000 if phase == 'A' else 100000)
            assert item['train']['lambda_a'] == strength
            assert item['train']['optim_g']['lr'] == 1e-4
            assert item['train']['rgb_lr'] == 1e-5
            assert item['path']['pretrain_kind'] == ('e0' if phase == 'A' else 'e3')
        test = yaml.safe_load((ROOT/f'options/test/mambairv2/test_E3_{tag}_x4.yml').read_text())
        assert test['datasets'] == test_source['datasets']
        assert test['val'] == test_source['val']
        assert test['scale'] == 4
    print('PASS: 12 E3 configs preserve all data paths/metrics and four lambda values')


def load_architectures():
    base = load_isolated((ROOT/'basicsr/archs/mambairv2_arch.py').read_text(encoding='utf-8'))
    e0 = load_isolated((ROOT/'basicsr/archs/udr_mambairv2_arch.py').read_text(encoding='utf-8'),
                       dict(MambaIRv2=base.MambaIRv2))
    e3 = load_isolated((ROOT/'basicsr/archs/e3_local_alpha_arch.py').read_text(encoding='utf-8'),
                       dict(DepthResidualExpert=e0.DepthResidualExpert,
                            UDRMambaIRv2=e0.UDRMambaIRv2))
    return base, e0, e3


def architecture_checks(e0, e3):
    kw = dict(img_size=8, embed_dim=12, depths=(1,) * 6,
              num_heads=(3,) * 6, window_size=4, d_state=2,
              inner_rank=4, num_tokens=4, mlp_ratio=1.,
              upscale=4, upsampler='pixelshuffle')
    old = e0.UDRMambaIRv2(**kw).eval()
    model = e3.E3LocalAlphaMambaIRv2(**kw).eval()
    model.load_e0_state_dict(old.state_dict())
    assert all(torch.equal(model.state_dict()[k], old.state_dict()[k])
               for k in old.state_dict())
    rgb, depth = torch.rand(1, 3, 8, 8), torch.rand(1, 1, 8, 8)
    torch.manual_seed(20)
    output = model(rgb, depth)
    alpha = model.udr.local_alpha_map
    assert output.shape == (1, 3, 32, 32)
    assert alpha.shape == (1, 1, 8, 8)
    assert .005 < alpha.mean().item() < .01
    assert alpha.min() > 0 and alpha.max() <= .1
    assert set(('local_alpha_mean', 'local_alpha_std', 'local_alpha_p05',
                'local_alpha_p50', 'local_alpha_p95', 'local_alpha_max',
                'correction_rms', 'correction_active_ratio')) <= set(model.udr_stats)
    _, maps, _ = infer_partitioned(model, rgb, depth, audit=True)
    torch.testing.assert_close(torch.from_numpy(maps['local_alpha']),
                               model.udr.local_alpha_map.squeeze().detach(),
                               rtol=1e-5, atol=1e-6)
    # With the new map clamped to E0's global alpha, the complete SR output
    # must match under the same hard-Gumbel seed. This isolates alpha alone.
    with torch.no_grad():
        last = model.udr.local_alpha_head.predictor[-1]
        last.weight.zero_()
        a = old.udr.alpha.item()
        last.bias.fill_(torch.logit(torch.tensor(a / model.udr.local_alpha_head.alpha_max)).item())
        torch.manual_seed(25)
        expected = old(rgb, depth)
        torch.manual_seed(25)
        actual = model(rgb, depth)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    model.configure_phase('A')
    assert not model.udr.alpha_raw.requires_grad
    assert all(parameter.requires_grad == name.startswith('udr.')
               for name, parameter in model.named_parameters()
               if name != 'udr.alpha_raw')
    model.configure_phase('B')
    assert all(parameter.requires_grad for name, parameter in model.named_parameters()
               if name != 'udr.alpha_raw')
    assert not model.udr.alpha_raw.requires_grad
    print('PASS: strict E0 transfer, spatial alpha, hook identity, constant-alpha E0 equivalence, A/B freeze')
    return kw, old


def training_checks(base, e3, kw, old):
    ModelBase = load_model_class(types.SimpleNamespace(UDRMambaIRv2=e3.E3LocalAlphaMambaIRv2), True)
    class Registry:
        def register(self):
            return lambda cls: cls
    m = load_isolated((ROOT/'basicsr/models/e3_local_alpha_model.py').read_text(encoding='utf-8'),
                      dict(UDRMambaIRv2Model=ModelBase,
                           SRModel=ModelBase.__mro__[1], MODEL_REGISTRY=Registry(),
                           get_root_logger=lambda: logging.getLogger('e3_check')))
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        old_path, a_path = folder/'e0.pth', folder/'a.pth'
        torch.save(dict(params=old.state_dict()), old_path)
        cfg = dict(is_train=True, dist=False, num_gpu=0,
                   network_g=dict(type='E3LocalAlphaMambaIRv2', local_alpha_init=.0075, **kw),
                   path=dict(pretrain_network_g=str(old_path), pretrain_kind='e0',
                             strict_load_g=True),
                   train=dict(phase='A', lambda_a=1e-4,
                              optim_g=dict(type='Adam', lr=1e-4),
                              scheduler=dict(type='MultiStepLR', milestones=[], gamma=1.),
                              pixel_opt=dict(type='L1Loss', loss_weight=1., reduction='mean')))
        sample = dict(lq=torch.rand(1, 3, 4, 4), depth=torch.rand(1, 1, 4, 4),
                      gt=torch.rand(1, 3, 16, 16))
        a = m.E3LocalAlphaMambaIRv2Model(deepcopy(cfg))
        before = {n: p.detach().clone() for n, p in a.get_bare_model(a.net_g).named_parameters()}
        a.feed_data(sample)
        a.optimize_parameters(1)
        net = a.get_bare_model(a.net_g)
        assert 'l_alpha' in a.log_dict
        for name, parameter in net.named_parameters():
            if name.startswith('udr.') and name != 'udr.alpha_raw':
                assert parameter.grad is not None, name
            else:
                assert parameter.grad is None, name
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        torch.save(dict(params=net.state_dict()), a_path)
        bcfg = deepcopy(cfg)
        bcfg['train'].update(phase='B', rgb_lr=1e-5)
        bcfg['path'].update(pretrain_network_g=str(a_path), pretrain_kind='e3')
        b = m.E3LocalAlphaMambaIRv2Model(bcfg)
        b.feed_data(sample)
        b.optimize_parameters(1)
        bnet = b.get_bare_model(b.net_g)
        assert bnet.conv_first.weight.grad is not None
        assert bnet.udr.projection.weight.grad is not None
        assert bnet.udr.local_alpha_head.predictor[0].weight.grad is not None
        assert bnet.udr.alpha_raw.grad is None
        assert [g['lr'] for g in b.optimizer_g.param_groups] == [1e-4, 1e-5]
    print('PASS: real synthetic A/B optimizer steps, local-alpha gradients, frozen legacy scalar, L1+sparse loss')


def main():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    check_configs()
    base, e0, e3 = load_architectures()
    kw, old = architecture_checks(e0, e3)
    training_checks(base, e3, kw, old)
    print('ALL E3 CPU REFERENCE CHECKS PASSED')


if __name__ == '__main__':
    main()
