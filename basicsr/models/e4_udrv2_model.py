"""E4 integration training: existing E2/E3 losses and two-phase LR policy."""
from copy import deepcopy

import torch
from torch.nn import functional as F

from basicsr.archs.mambairv2_arch import MambaIRv2
from basicsr.models.e2_udr_mambairv2_model import normalized_rgb_error, pearson_map
from basicsr.models.sr_model import SRModel
from basicsr.models.udr_mambairv2_model import UDRMambaIRv2Model
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class E4UDRMambaIRv2Model(UDRMambaIRv2Model):
    def __init__(self, opt):
        if opt['is_train']:
            train = opt['train']
            mode = opt['network_g']['uncertainty_mode']
            expected_u = .01 if mode == 'learned_error' else 0.
            if float(train.get('lambda_u', -1)) != expected_u:
                raise ValueError('E4 must preserve the selected E2 uncertainty loss.')
            if float(train.get('lambda_a', -1)) not in (0., 1e-5, 1e-4, 5e-4):
                raise ValueError('E4 must preserve the selected E3 lambda_a.')
            if mode == 'learned_error' and not train.get('rgb_teacher_checkpoint'):
                raise ValueError('E4 U3 requires the fixed RGB-only teacher checkpoint.')
            if opt['path'].get('pretrain_kind') != 'e4':
                raise ValueError('E4 trains only from a strictly merged E4 checkpoint.')
        super().__init__(opt)
        if self.is_train and opt['network_g']['uncertainty_mode'] == 'learned_error':
            # The same immutable RGB teacher/target as E2-U3; it stays outside
            # net_g and the E4 optimizer in both phases.
            options = deepcopy(opt['network_g'])
            for key in ('type', 'uncertainty_mode', 'local_alpha_init',
                        'alpha_max', 'alpha_init', 'projection_std'):
                options.pop(key, None)
            self.rgb_teacher = MambaIRv2(**options).to(self.device).eval()
            state = torch.load(opt['train']['rgb_teacher_checkpoint'], map_location='cpu')
            if 'params' not in state:
                raise ValueError('RGB teacher checkpoint lacks params.')
            weights = {(key[7:] if key.startswith('module.') else key): value
                       for key, value in state['params'].items()}
            self.rgb_teacher.load_state_dict(weights, strict=True)
            for parameter in self.rgb_teacher.parameters():
                parameter.requires_grad_(False)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        if not strict:
            raise ValueError('E4 requires a full strictly merged checkpoint.')
        return SRModel.load_network(self, net, load_path, strict=True,
                                    param_key=param_key)

    def setup_optimizers(self):
        train = self.opt['train']
        settings = deepcopy(train['optim_g'])
        optim_type, adaptive_lr = settings.pop('type'), settings.pop('lr')
        rgb_lr = train.get('rgb_lr', adaptive_lr * .1)
        adaptive, rgb = [], []
        for name, parameter in self.get_bare_model(self.net_g).named_parameters():
            if parameter.requires_grad:
                (adaptive if name.startswith(('udr.', 'uncertainty_head.'))
                 else rgb).append(parameter)
        if not adaptive or (train['phase'] == 'A' and rgb) or (train['phase'] == 'B' and not rgb):
            raise RuntimeError('E4 optimizer groups do not match Phase A/B freeze policy.')
        groups = [dict(params=adaptive, lr=adaptive_lr, group_name='adaptive')]
        if train['phase'] == 'B':
            groups.append(dict(params=rgb, lr=rgb_lr, group_name='rgb'))
        self.optimizer_g = self.get_optimizer(optim_type, groups,
                                              lr=adaptive_lr, **settings)
        self.optimizers.append(self.optimizer_g)
        get_root_logger().info('E4 Phase %s groups: %s', train['phase'],
                               [(g['group_name'], g['lr']) for g in groups])

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad(set_to_none=True)
        self.output = self.net_g(self.lq, self.depth)
        net = self.get_bare_model(self.net_g)
        l_pix = self.cri_pix(self.output, self.gt)
        l_alpha = net.udr.local_alpha_map.abs().mean()
        loss = l_pix + float(self.opt['train']['lambda_a']) * l_alpha
        stats = dict(l_pix=l_pix, l_alpha=l_alpha, **net.udr_stats)
        if net.uncertainty_mode == 'learned_error':
            with torch.no_grad():
                target = normalized_rgb_error(self.rgb_teacher(self.lq), self.gt)
            if net.uncertainty_map.shape != target.shape:
                raise RuntimeError('E4 uncertainty and RGB teacher error are misaligned.')
            l_u = F.l1_loss(net.uncertainty_map, target)
            loss = loss + float(self.opt['train']['lambda_u']) * l_u
            stats.update(l_u=l_u, uncertainty_rgb_error_pearson=pearson_map(
                net.uncertainty_map, target))
        loss.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(stats)
