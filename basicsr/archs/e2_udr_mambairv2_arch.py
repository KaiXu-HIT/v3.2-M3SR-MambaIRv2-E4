"""E2: change only the spatial RGB uncertainty used by the v3.2 UDR gate.

U1 is pre-Gumbel routing concentration, U2 is local RGB-feature variance,
and U3 is a lightweight learned head supervised by frozen RGB-only errors.
The Depth expert, global alpha, backbone operations, and SR head stay intact.
"""
import torch
from torch import nn
from torch.nn import functional as F

from basicsr.archs.udr_mambairv2_arch import UDRMambaIRv2
from basicsr.utils.registry import ARCH_REGISTRY


class ReconstructionUncertaintyHead(nn.Sequential):
    """E2-U3: Conv3x3(C,C//4) -> GELU -> DWConv3x3 -> GELU -> Conv1x1 -> sigmoid."""
    def __init__(self, channels):
        hidden = max(1, channels // 4)
        super().__init__(nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(),
                         nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                         nn.GELU(), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())


def local_feature_uncertainty(feature):
    """E2-U2: per-channel 3x3 spatial variance with per-image robust scaling."""
    mean = F.avg_pool2d(F.pad(feature.float(), (1, 1, 1, 1), mode='replicate'), 3, 1)
    square = F.avg_pool2d(F.pad(feature.float().square(), (1, 1, 1, 1),
                                mode='replicate'), 3, 1)
    variance = (square - mean.square()).clamp_min(0).mean(1, keepdim=True)
    scale = torch.quantile(variance.detach().flatten(2), .95, dim=-1)
    return (variance / scale[..., None, None].clamp_min(1e-6)).clamp(0, 1).to(feature)


@ARCH_REGISTRY.register()
class E2UDRMambaIRv2(UDRMambaIRv2):
    MODES = ('route_concentration', 'feature_variance', 'learned_error')

    def __init__(self, uncertainty_mode='learned_error', **kwargs):
        if uncertainty_mode not in self.MODES:
            raise ValueError('E2 uncertainty_mode must be U1, U2 or U3.')
        super().__init__(**kwargs)
        self.uncertainty_mode = uncertainty_mode
        if uncertainty_mode == 'learned_error':
            # Only E2-U3 adds trainable weights; all original checkpoint keys
            # retain their names and shapes for strict E0 transfer.
            self.uncertainty_head = ReconstructionUncertaintyHead(self.embed_dim)
        self.e2_phase = 'B'
        self.uncertainty_map = None

    def configure_phase(self, phase):
        if phase not in ('A', 'B'):
            raise ValueError('E2 phase must be A or B.')
        if phase == 'A' and self.uncertainty_mode != 'learned_error':
            raise ValueError('Only learned E2-U3 has trainable Phase A weights.')
        self.e2_phase = phase
        for name, parameter in self.named_parameters():
            # Phase A freezes both RGB and the complete existing Depth branch.
            parameter.requires_grad_(phase == 'B' or name.startswith('uncertainty_head.'))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, 'e2_phase', None) == 'A':
            for name, child in self.named_children():
                if name != 'uncertainty_head':
                    child.eval()
        return self

    def load_e0_state_dict(self, state):
        """Require every E0 tensor; initialize only the U3 uncertainty head."""
        own = self.state_dict()
        expected = {key for key in own if not key.startswith('uncertainty_head.')}
        missing, unexpected = expected - set(state), set(state) - expected
        if missing or unexpected:
            raise RuntimeError(f'E0 transfer mismatch: missing={sorted(missing)}, '
                               f'unexpected={sorted(unexpected)}')
        for key in expected:
            if own[key].shape != state[key].shape:
                raise RuntimeError(f'E0 tensor shape mismatch: {key}')
        own.update(state)
        self.load_state_dict(own, strict=True)

    def forward_features(self, x, params):
        size = x.shape[-2:]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        route_maps = []
        for stage, layer in enumerate(self.layers, start=1):
            # E2-U1 reads routing probabilities only. Depth never enters ASSM.
            stage_params = dict(attn_mask=params['attn_mask'], rpi_sa=params['rpi_sa'])
            if self.uncertainty_mode == 'route_concentration' and stage >= 4:
                collector = []
                stage_params.update(ambiguity_collector=collector,
                                    routing_uncertainty_mode='concentration')
            x = layer(x, size, stage_params)
            if self.uncertainty_mode == 'route_concentration' and stage >= 4:
                route_maps.append(torch.stack(collector).mean(0))
        feature = self.patch_unembed(self.norm(x), size)
        if self.uncertainty_mode == 'route_concentration':
            uncertainty = torch.stack(route_maps).mean(0)
        elif self.uncertainty_mode == 'feature_variance':
            uncertainty = local_feature_uncertainty(feature)
        else:
            uncertainty = self.uncertainty_head(feature)
        self.uncertainty_map = uncertainty
        # The unchanged DepthResidualExpert consumes uncertainty at the same
        # single late fusion point; no other reconstruction path is modified.
        feature, stats = self.udr(feature, params['rgb'], params['depth'], uncertainty)
        u = uncertainty.detach().float().flatten()
        stats.update(uncertainty_mean=u.mean(),
                     uncertainty_std=u.std(unbiased=False),
                     uncertainty_p05=torch.quantile(u, .05),
                     uncertainty_p95=torch.quantile(u, .95))
        self.udr_stats = stats
        return feature
