"""E2 training: frozen RGB-error teacher, U3-only Phase A, joint Phase B."""
from copy import deepcopy

import torch
from torch.nn import functional as F

from basicsr.archs.mambairv2_arch import MambaIRv2
from basicsr.models.sr_model import SRModel
from basicsr.models.udr_mambairv2_model import UDRMambaIRv2Model
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


def normalized_rgb_error(rgb_prediction, ground_truth, scale=4):
    """E2-U3 target: P95-normalized fixed RGB-only error, aligned at LR pixels."""
    if rgb_prediction.shape != ground_truth.shape:
        raise ValueError('RGB teacher prediction and GT must have identical shape.')
    if rgb_prediction.shape[-2] % scale or rgb_prediction.shape[-1] % scale:
        raise ValueError('Teacher error must divide exactly into LR cells.')
    error = (rgb_prediction.detach() - ground_truth).abs().mean(1, keepdim=True)
    error = F.avg_pool2d(error, kernel_size=scale, stride=scale)
    p95 = torch.quantile(error.flatten(1), .95, dim=1)
    return (error / p95[:, None, None, None].clamp_min(1e-6)).clamp(0, 1)


def pearson_map(a, b):
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return torch.where(den > 1e-8, (a * b).sum() / den.clamp_min(1e-8),
                       torch.zeros_like(den))


@MODEL_REGISTRY.register()
class E2UDRMambaIRv2Model(UDRMambaIRv2Model):
    def __init__(self, opt):
        if opt['is_train']:
            train = opt['train']
            if train['phase'] == 'A' and opt['network_g']['uncertainty_mode'] != 'learned_error':
                raise ValueError('E2 Phase A applies only to learned U3.')
            if train.get('lambda_u', 0) != (0.01 if opt['network_g']['uncertainty_mode'] == 'learned_error' else 0):
                raise ValueError('E2 uses lambda_u=0.01 for U3 and 0 for U1/U2.')
            if not train.get('rgb_teacher_checkpoint'):
                raise ValueError('E2 requires the fixed RGB-only baseline checkpoint.')
        super().__init__(opt)
        if self.is_train:
            # The teacher is outside net_g and the optimizer. It is loaded
            # strictly from the fixed E0 RGB baseline, never updated in A/B.
            options = deepcopy(opt['network_g'])
            for key in ('type', 'uncertainty_mode', 'alpha_max', 'alpha_init',
                        'projection_std'):
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
        # A starts from E0 Phase B, whose only absent E2 weights are the U3
        # head. Same-phase resume and E2 Phase B require full strict E2 state.
        kind = ('e2' if self.opt['path'].get('resume_state') else
                self.opt['path'].get('pretrain_kind', 'e2'))
        if kind == 'e2':
            return SRModel.load_network(self, net, load_path, strict=True,
                                        param_key=param_key)
        if kind != 'e0' or not strict:
            raise ValueError('E2 accepts only strict E0 transfer or full E2 checkpoints.')
        state = torch.load(load_path, map_location='cpu')
        if param_key not in state:
            raise KeyError(f'E0 checkpoint lacks {param_key}.')
        weights = {(key[7:] if key.startswith('module.') else key): value
                   for key, value in state[param_key].items()}
        self.get_bare_model(net).load_e0_state_dict(weights)
        get_root_logger().info('E2: strictly transferred every E0 tensor; only U3 head is new.')

    def setup_optimizers(self):
        train = self.opt['train']
        settings = deepcopy(train['optim_g'])
        optim_type, depth_lr = settings.pop('type'), settings.pop('lr')
        rgb_lr = train.get('rgb_lr', depth_lr * .1)
        head, depth, rgb = [], [], []
        for name, parameter in self.get_bare_model(self.net_g).named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('uncertainty_head.'):
                head.append(parameter)
            elif name.startswith('udr.'):
                depth.append(parameter)
            else:
                rgb.append(parameter)
        if train['phase'] == 'A':
            if not head or depth or rgb:
                raise RuntimeError('E2 Phase A must train only the U3 head.')
            groups = [dict(params=head, lr=depth_lr, group_name='uncertainty')]
        else:
            if not depth or not rgb:
                raise RuntimeError('E2 Phase B requires Depth and RGB parameters.')
            groups = [dict(params=depth, lr=depth_lr, group_name='depth')]
            if head:
                groups.append(dict(params=head, lr=depth_lr, group_name='uncertainty'))
            groups.append(dict(params=rgb, lr=rgb_lr, group_name='rgb'))
        self.optimizer_g = self.get_optimizer(optim_type, groups, lr=depth_lr, **settings)
        self.optimizers.append(self.optimizer_g)
        get_root_logger().info('E2 Phase %s groups: %s', train['phase'],
                               [(g['group_name'], g['lr']) for g in groups])

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad(set_to_none=True)
        self.output = self.net_g(self.lq, self.depth)
        net = self.get_bare_model(self.net_g)
        l_pix = self.cri_pix(self.output, self.gt)
        loss = l_pix
        with torch.no_grad():
            # The frozen teacher supplies a fixed RGB-only error target for
            # all E2 variants; it never sees Depth or UDR correction.
            teacher_sr = self.rgb_teacher(self.lq)
            target = normalized_rgb_error(teacher_sr, self.gt)
        uncertainty = net.uncertainty_map
        if uncertainty.shape != target.shape:
            raise RuntimeError('E2 uncertainty and RGB-error target are misaligned.')
        if net.uncertainty_mode == 'learned_error':
            l_u = F.l1_loss(uncertainty, target)
            loss = loss + self.opt['train']['lambda_u'] * l_u
        loss.backward()
        self.optimizer_g.step()
        stats = dict(l_pix=l_pix, **net.udr_stats,
                     uncertainty_rgb_error_pearson=pearson_map(uncertainty, target))
        if net.uncertainty_mode == 'learned_error':
            stats['l_u'] = l_u.detach()
        interval = int(self.opt['train'].get('mechanism_log_interval', 500))
        if current_iter % interval == 0:
            # Spearman is diagnostic only; detached CPU ranks cannot affect
            # the gradient or disturb the RGB/Depth control variables.
            from scipy.stats import spearmanr
            u = uncertainty.detach().float().cpu().numpy().ravel()
            t = target.detach().float().cpu().numpy().ravel()
            rho = spearmanr(u, t).statistic
            value = float(rho) if rho == rho else 0.0
            stats['uncertainty_rgb_error_spearman'] = uncertainty.new_tensor(value)
        self.log_dict = self.reduce_loss_dict(stats)
