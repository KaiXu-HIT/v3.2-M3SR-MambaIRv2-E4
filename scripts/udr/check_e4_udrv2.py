"""Offline E4 contracts using the existing differentiable CPU scan reference."""
import ast
from copy import deepcopy
import logging
from pathlib import Path
import sys
import tempfile
import types

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.grs.check_grs import load_isolated
from scripts.udr.check_udr import load_model_class
from scripts.udr.e0_mechanism_audit import infer_partitioned
from scripts.udr.prepare_e4_integration import merge_parameters


def load_architectures():
    def source(path):
        return (ROOT/path).read_text(encoding='utf-8')
    base = load_isolated(source('basicsr/archs/mambairv2_arch.py'))
    e0 = load_isolated(source('basicsr/archs/udr_mambairv2_arch.py'),
                       dict(MambaIRv2=base.MambaIRv2))
    e2 = load_isolated(source('basicsr/archs/e2_udr_mambairv2_arch.py'),
                       dict(UDRMambaIRv2=e0.UDRMambaIRv2))
    e3 = load_isolated(source('basicsr/archs/e3_local_alpha_arch.py'),
                       dict(DepthResidualExpert=e0.DepthResidualExpert,
                            UDRMambaIRv2=e0.UDRMambaIRv2))
    e4 = load_isolated(source('basicsr/archs/e4_udrv2_arch.py'),
                       dict(E2UDRMambaIRv2=e2.E2UDRMambaIRv2,
                            LocalAlphaDepthExpert=e3.LocalAlphaDepthExpert))
    return base, e0, e2, e3, e4


def architecture_checks(e2, e3, e4):
    kw = dict(img_size=8, embed_dim=12, depths=(1,) * 6,
              num_heads=(3,) * 6, window_size=4, d_state=2,
              inner_rank=4, num_tokens=4, mlp_ratio=1.,
              upscale=4, upsampler='pixelshuffle')
    rgb, depth = torch.rand(1, 3, 8, 8), torch.rand(1, 1, 8, 8)
    e3_model = e3.E3LocalAlphaMambaIRv2(**kw)
    for variant, mode in (('U1', 'route_concentration'),
                          ('U2', 'feature_variance'), ('U3', 'learned_error')):
        e2_model = e2.E2UDRMambaIRv2(uncertainty_mode=mode, **kw)
        model = e4.E4UDRMambaIRv2(uncertainty_mode=mode, **kw).eval()
        merged = merge_parameters(e2_model.state_dict(), e3_model.state_dict(), variant)
        model.load_state_dict(merged, strict=True)
        assert set(model.state_dict()) == set(merged)
        torch.manual_seed(42)
        output, maps, _ = infer_partitioned(model, rgb, depth, audit=True)
        assert output.shape == (1, 3, 32, 32)
        assert maps['ambiguity'].shape == (8, 8)
        assert maps['local_alpha'].shape == (8, 8)
        assert maps['confidence'].shape == (8, 8)
        assert maps['correction_abs'].shape == (8, 8)
        assert 0 < maps['local_alpha'].min() < maps['local_alpha'].max() < .1
        model.configure_phase('A')
        assert all(parameter.requires_grad ==
                   (name.startswith(('udr.', 'uncertainty_head.'))
                    and name != 'udr.alpha_raw')
                   for name, parameter in model.named_parameters())
        model.train()
        assert model.udr.training and not model.layers[0].training
        if variant == 'U3':
            assert model.uncertainty_head.training
        model.configure_phase('B')
        assert all(parameter.requires_grad == (name != 'udr.alpha_raw')
                   for name, parameter in model.named_parameters())
    print('PASS: U1/U2/U3 strict E2+E3 merge, four-map product hooks and A/B freeze')
    return kw


def extract_e2_functions():
    source = (ROOT/'basicsr/models/e2_udr_mambairv2_model.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('normalized_rgb_error', 'pearson_map')]
    from torch.nn import functional as F
    scope = dict(torch=torch, F=F)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<e2 functions>', 'exec'), scope)
    return scope


def training_checks(base, e4, kw):
    BaseModel = load_model_class(types.SimpleNamespace(UDRMambaIRv2=e4.E4UDRMambaIRv2), True)
    class Registry:
        def register(self):
            return lambda cls: cls
    helper = extract_e2_functions()
    model_module = load_isolated((ROOT/'basicsr/models/e4_udrv2_model.py').read_text(encoding='utf-8'),
                                 dict(UDRMambaIRv2Model=BaseModel,
                                      MambaIRv2=base.MambaIRv2,
                                      SRModel=BaseModel.__mro__[1],
                                      MODEL_REGISTRY=Registry(), deepcopy=deepcopy,
                                      normalized_rgb_error=helper['normalized_rgb_error'],
                                      pearson_map=helper['pearson_map'],
                                      get_root_logger=lambda: logging.getLogger('e4_check')))
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        teacher = base.MambaIRv2(**kw)
        teacher_path, init_path, a_path = (folder/name for name in
                                           ('rgb.pth', 'e4_init.pth', 'e4_a.pth'))
        torch.save(dict(params=teacher.state_dict()), teacher_path)
        init = e4.E4UDRMambaIRv2(uncertainty_mode='learned_error', **kw)
        torch.save(dict(params=init.state_dict()), init_path)
        cfg = dict(is_train=True, dist=False, num_gpu=0,
                   network_g=dict(type='E4UDRMambaIRv2', uncertainty_mode='learned_error', **kw),
                   path=dict(pretrain_network_g=str(init_path), pretrain_kind='e4',
                             strict_load_g=True),
                   train=dict(phase='A', lambda_a=1e-4, lambda_u=.01,
                              rgb_teacher_checkpoint=str(teacher_path),
                              optim_g=dict(type='Adam', lr=1e-4),
                              scheduler=dict(type='MultiStepLR', milestones=[], gamma=1.),
                              pixel_opt=dict(type='L1Loss', loss_weight=1., reduction='mean')))
        sample = dict(lq=torch.rand(1, 3, 4, 4), depth=torch.rand(1, 1, 4, 4),
                      gt=torch.rand(1, 3, 16, 16))
        a = model_module.E4UDRMambaIRv2Model(deepcopy(cfg))
        before = {n: p.detach().clone() for n, p in a.get_bare_model(a.net_g).named_parameters()}
        a.feed_data(sample)
        a.optimize_parameters(1)
        net = a.get_bare_model(a.net_g)
        assert 'l_alpha' in a.log_dict and 'l_u' in a.log_dict
        for name, parameter in net.named_parameters():
            if name.startswith(('udr.', 'uncertainty_head.')) and name != 'udr.alpha_raw':
                assert parameter.grad is not None, name
            else:
                assert parameter.grad is None, name
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        assert [g['lr'] for g in a.optimizer_g.param_groups] == [1e-4]
        torch.save(dict(params=net.state_dict()), a_path)
        bcfg = deepcopy(cfg)
        bcfg['train'].update(phase='B', rgb_lr=1e-5)
        bcfg['path']['pretrain_network_g'] = str(a_path)
        b = model_module.E4UDRMambaIRv2Model(bcfg)
        b.feed_data(sample)
        b.optimize_parameters(1)
        bnet = b.get_bare_model(b.net_g)
        assert bnet.conv_first.weight.grad is not None
        assert bnet.udr.projection.weight.grad is not None
        assert bnet.uncertainty_head[0].weight.grad is not None
        assert bnet.udr.local_alpha_head.predictor[0].weight.grad is not None
        assert bnet.udr.alpha_raw.grad is None
        assert [g['lr'] for g in b.optimizer_g.param_groups] == [1e-4, 1e-5]
    print('PASS: U3 E4 real synthetic A/B optimizer steps, U+alpha loss and strict checkpoint')


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    base, e0, e2, e3, e4 = load_architectures()
    kw = architecture_checks(e2, e3, e4)
    training_checks(base, e4, kw)
    print('ALL E4 CPU REFERENCE CHECKS PASSED')


if __name__ == '__main__':
    main()
