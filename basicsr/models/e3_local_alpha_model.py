"""E3 two-phase optimization and sparse regularization of local alpha only."""
import torch

from basicsr.models.sr_model import SRModel
from basicsr.models.udr_mambairv2_model import UDRMambaIRv2Model
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class E3LocalAlphaMambaIRv2Model(UDRMambaIRv2Model):
    def __init__(self, opt):
        if opt['is_train']:
            value = float(opt['train'].get('lambda_a', -1))
            if value not in (0.0, 1e-5, 1e-4, 5e-4):
                raise ValueError('E3 lambda_a must be one of 0, 1e-5, 1e-4, 5e-4.')
        super().__init__(opt)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        kind = ('e3' if self.opt['path'].get('resume_state') else
                self.opt['path'].get('pretrain_kind', 'e3'))
        if kind == 'e3':
            return SRModel.load_network(self, net, load_path, strict=True,
                                        param_key=param_key)
        if kind != 'e0' or not strict or not self.is_train or self.opt['train']['phase'] != 'A':
            raise ValueError('Only E3 Phase A may strictly transfer an E0 checkpoint.')
        state = torch.load(load_path, map_location='cpu')
        if param_key not in state:
            raise KeyError(f'E0 checkpoint lacks {param_key}.')
        weights = {(key[7:] if key.startswith('module.') else key): value
                   for key, value in state[param_key].items()}
        self.get_bare_model(net).load_e0_state_dict(weights)
        get_root_logger().info('E3: strictly loaded all E0 tensors; only local-alpha head is new.')

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad(set_to_none=True)
        self.output = self.net_g(self.lq, self.depth)
        net = self.get_bare_model(self.net_g)
        l_pix = self.cri_pix(self.output, self.gt)
        # E3 changes only the alpha mechanism. The original L1 reconstruction
        # objective stays intact; this optional term sparsifies the 1×H×W map.
        l_alpha = net.udr.local_alpha_map.abs().mean()
        loss = l_pix + float(self.opt['train']['lambda_a']) * l_alpha
        loss.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(dict(l_pix=l_pix, l_alpha=l_alpha,
                                                   **net.udr_stats))
