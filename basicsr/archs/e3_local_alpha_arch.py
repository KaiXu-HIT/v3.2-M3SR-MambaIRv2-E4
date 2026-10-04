"""E3: replace only the global UDR correction strength with a spatial map.

The E0 RGB entropy, Depth encoder, confidence gate, residual projection,
backbone and reconstruction head are reused unchanged. E4 will combine the
best E2 uncertainty with this mechanism; E3 deliberately does not do so.
"""
import math

import torch
from torch import nn

from basicsr.archs.udr_mambairv2_arch import DepthResidualExpert, UDRMambaIRv2
from basicsr.utils.registry import ARCH_REGISTRY


class LocalAlphaHead(nn.Module):
    """One-channel alpha from compressed RGB/Depth features, gradient, confidence."""
    def __init__(self, channels, alpha_max=.1, alpha_init=.0075):
        super().__init__()
        if not 0 < alpha_init < alpha_max <= .1:
            raise ValueError('Require 0 < local alpha init < alpha max <= 0.1.')
        hidden = max(8, channels // 8)
        self.rgb_compress = nn.Conv2d(channels, hidden, 1)
        self.depth_compress = nn.Conv2d(32, hidden, 1)
        self.predictor = nn.Sequential(
            nn.Conv2d(2 * hidden + 2, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, 1, 1))
        # Initialize E[A_D] near 0.0075, inside the specified 0.005–0.01 band.
        # A tiny nonzero final weight keeps gradients to earlier layers on step 1.
        nn.init.normal_(self.predictor[-1].weight, std=1e-3)
        nn.init.constant_(self.predictor[-1].bias,
                          math.log(alpha_init / (alpha_max - alpha_init)))
        self.alpha_max = float(alpha_max)

    def forward(self, feature, geometry, depth_gradient, confidence):
        context = torch.cat((self.rgb_compress(feature),
                             self.depth_compress(geometry),
                             depth_gradient, confidence), dim=1)
        return self.alpha_max * self.predictor(context).sigmoid()


class LocalAlphaDepthExpert(DepthResidualExpert):
    """E0 expert with identical Depth computation and a spatial alpha multiplier."""
    def __init__(self, channels, alpha_max=.1, alpha_init=.01,
                 projection_std=1e-3, local_alpha_init=.0075):
        super().__init__(channels, alpha_max, alpha_init, projection_std)
        self.local_alpha_head = LocalAlphaHead(channels, alpha_max, local_alpha_init)
        self.local_alpha_map = None
        # alpha_raw is retained only so every E0 checkpoint tensor transfers
        # strictly; E3 never uses it in the correction and never optimizes it.
        self.alpha_raw.requires_grad_(False)

    def forward(self, feature, rgb, depth, ambiguity):
        ed = self.gradient(depth)
        er = self.gradient((rgb * self.luma.to(rgb)).sum(1, keepdim=True))
        confidence = self.gre(torch.cat((er, ed, (er - ed).abs(), er * ed), 1))
        geometry = self.encoder(torch.cat((depth, ed), 1))
        context = self.context_norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        residual = self.projection(self.fusion_dw(torch.cat((context, geometry), 1)))
        gate = ambiguity.to(feature) * confidence
        local_alpha = self.local_alpha_head(feature, geometry, ed, confidence)
        self.local_alpha_map = local_alpha
        correction = local_alpha * gate * residual
        a = local_alpha.detach().float().flatten()
        stats = dict(ambiguity_mean=ambiguity.detach().mean(),
                     confidence_mean=confidence.detach().mean(),
                     gate_mean=gate.detach().mean(),
                     gate_std=gate.detach().float().std(unbiased=False),
                     local_alpha_mean=a.mean(),
                     local_alpha_std=a.std(unbiased=False),
                     local_alpha_p05=torch.quantile(a, .05),
                     local_alpha_p50=torch.quantile(a, .50),
                     local_alpha_p95=torch.quantile(a, .95),
                     local_alpha_max=a.max(),
                     correction_rms=correction.detach().square().mean().sqrt(),
                     correction_active_ratio=(a > .01).float().mean())
        return feature + correction, stats


@ARCH_REGISTRY.register()
class E3LocalAlphaMambaIRv2(UDRMambaIRv2):
    def __init__(self, local_alpha_init=.0075, **kwargs):
        super().__init__(**kwargs)
        self.udr = LocalAlphaDepthExpert(self.embed_dim,
                                         alpha_max=kwargs.get('alpha_max', .1),
                                         alpha_init=kwargs.get('alpha_init', .01),
                                         projection_std=kwargs.get('projection_std', 1e-3),
                                         local_alpha_init=local_alpha_init)

    def configure_phase(self, phase):
        super().configure_phase(phase)
        # Both phases train the local predictor and Depth branch; RGB is frozen
        # only in A. The unused E0 scalar remains a frozen compatibility tensor.
        self.udr.alpha_raw.requires_grad_(False)

    def load_e0_state_dict(self, state):
        """Strict E0 transfer: only new local-alpha-head keys may be absent."""
        own = self.state_dict()
        expected = {key for key in own if not key.startswith('udr.local_alpha_head.')}
        missing, unexpected = expected - set(state), set(state) - expected
        if missing or unexpected:
            raise RuntimeError(f'E0 transfer mismatch: missing={sorted(missing)}, '
                               f'unexpected={sorted(unexpected)}')
        for key in expected:
            if own[key].shape != state[key].shape:
                raise RuntimeError(f'E0 checkpoint tensor shape mismatch: {key}')
        own.update(state)
        self.load_state_dict(own, strict=True)
