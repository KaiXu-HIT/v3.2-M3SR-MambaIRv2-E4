"""E4 UDR-v2: compose the selected E2 uncertainty and E3 local alpha.

There are no new feature, confidence, routing, Depth, or reconstruction
modules. The only combination is F + A_D * U_R * C_D * R_D at the existing
late residual point. Selection and strict checkpoint merge are separate.
"""
from torch import nn

from basicsr.archs.e2_udr_mambairv2_arch import E2UDRMambaIRv2
from basicsr.archs.e3_local_alpha_arch import LocalAlphaDepthExpert
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class E4UDRMambaIRv2(E2UDRMambaIRv2):
    def __init__(self, local_alpha_init=.0075, **kwargs):
        super().__init__(**kwargs)
        # Reuse the exact E3 expert implementation; inherited E2
        # forward_features supplies the chosen U_R map as its gate input.
        self.udr = LocalAlphaDepthExpert(
            self.embed_dim, alpha_max=kwargs.get('alpha_max', .1),
            alpha_init=kwargs.get('alpha_init', .01),
            projection_std=kwargs.get('projection_std', 1e-3),
            local_alpha_init=local_alpha_init)
        self.e4_phase = 'B'

    def configure_phase(self, phase):
        if phase not in ('A', 'B'):
            raise ValueError('E4 phase must be A or B.')
        self.e4_phase = phase
        for name, parameter in self.named_parameters():
            # A adapts the imported E2 U head (if learned) and E3 Depth/local
            # head to each other. B additionally unfreezes the RGB backbone.
            train = phase == 'B' or name.startswith(('udr.', 'uncertainty_head.'))
            parameter.requires_grad_(train and name != 'udr.alpha_raw')
        self.train(self.training)

    def train(self, mode=True):
        # Bypass E2's U3-only Phase A runtime rule: E4 Phase A also trains
        # Depth/local alpha. Frozen RGB children remain in eval mode.
        nn.Module.train(self, mode)
        if mode and getattr(self, 'e4_phase', None) == 'A':
            for name, child in self.named_children():
                if name not in ('udr', 'uncertainty_head'):
                    child.eval()
        return self
