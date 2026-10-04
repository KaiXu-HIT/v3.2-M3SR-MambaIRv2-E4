"""UDR v3.2: read-only RGB routing ambiguity + reliable late depth residual.

The original RGB parameter names and operations are retained. No depth is sent
to routing, sorting, Delta, A/B/C, or selective scan. All new weights live in udr.*.
Depth is P2/P98-normalized on the full LR image by RGBDepthPairedImageDataset;
do not normalize it again after random cropping or inference partitioning.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from basicsr.archs.mambairv2_arch import MambaIRv2
from basicsr.utils.registry import ARCH_REGISTRY


class SpatialGradient(nn.Module):
    """Fixed Sobel magnitude, shared scale for RGB luminance and normalized depth."""
    def __init__(self):
        super().__init__()
        k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer('kernels', torch.stack([k, k.t()]).unsqueeze(1))

    def forward(self, x):
        edges = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), self.kernels.to(x))
        return torch.linalg.vector_norm(edges, dim=1, keepdim=True)


class DepthResidualExpert(nn.Module):
    def __init__(self, channels, alpha_max=0.1, alpha_init=0.01, projection_std=1e-3):
        super().__init__()
        if not 0 < alpha_max <= 0.1 or not 0 < abs(alpha_init) < alpha_max:
            raise ValueError('Require 0 < |alpha_init| < alpha_max <= 0.1.')
        if not 0 < projection_std <= 0.01:
            raise ValueError('Use a small positive residual projection std, at most 0.01.')
        self.alpha_max = float(alpha_max)
        # Nonzero alpha + small nonzero projection lets both branches learn on step 1.
        self.alpha_raw = nn.Parameter(torch.tensor(math.atanh(alpha_init / alpha_max)))
        self.gradient = SpatialGradient()
        self.register_buffer('luma', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.gre = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 8, 3, padding=1), nn.GELU(),
            nn.Conv2d(8, 1, 1), nn.Sigmoid())
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.GELU(),
            nn.Conv2d(32, 32, 1))
        self.context_norm = nn.LayerNorm(channels)
        self.fusion_dw = nn.Conv2d(channels + 32, channels + 32, 3,
                                   padding=1, groups=channels + 32)
        self.projection = nn.Conv2d(channels + 32, channels, 1)
        nn.init.normal_(self.projection.weight, std=projection_std)
        nn.init.zeros_(self.projection.bias)

    @property
    def alpha(self):
        return self.alpha_max * self.alpha_raw.tanh()

    def forward(self, feature, rgb, depth, ambiguity):
        ed = self.gradient(depth)
        er = self.gradient((rgb * self.luma.to(rgb)).sum(1, keepdim=True))
        confidence = self.gre(torch.cat([er, ed, (er - ed).abs(), er * ed], 1))
        geometry = self.encoder(torch.cat([depth, ed], 1))
        context = self.context_norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        residual = self.projection(self.fusion_dw(torch.cat([context, geometry], 1)))
        gate = ambiguity.to(feature) * confidence
        correction = self.alpha * gate * residual
        # Diagnostics retain scalars only; never store tensors with an autograd graph.
        stats = dict(ambiguity_mean=ambiguity.detach().mean(),
                     confidence_mean=confidence.detach().mean(), gate_mean=gate.detach().mean(),
                     # E2 mechanism log only; the gate calculation is unchanged.
                     gate_std=gate.detach().float().std(unbiased=False),
                     alpha=self.alpha.detach(), correction_rms=correction.detach().square().mean().sqrt())
        return feature + correction, stats


@ARCH_REGISTRY.register()
class UDRMambaIRv2(MambaIRv2):
    def __init__(self, alpha_max=0.1, alpha_init=0.01, projection_std=1e-3, **kwargs):
        defaults = dict(in_chans=3, upscale=4, upsampler='pixelshuffle',
                        embed_dim=174, depths=(6,) * 6, num_heads=(6,) * 6,
                        d_state=16, inner_rank=64, num_tokens=128)
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        if kwargs['in_chans'] != 3 or kwargs['upscale'] != 4 or kwargs['upsampler'] != 'pixelshuffle':
            raise ValueError('UDR requires the original classical RGB x4 pixelshuffle head.')
        if len(kwargs['depths']) != 6 or any(d < 1 for d in kwargs['depths']):
            raise ValueError('UDR requires six nonempty ASSBs; ambiguity is read from stages 4/5/6.')
        if kwargs['num_tokens'] < 2:
            raise ValueError('Normalized routing entropy requires at least two tokens.')
        super().__init__(**kwargs)
        # Attach only after RGB initialization to retain RGB state-dict compatibility.
        self.udr = DepthResidualExpert(self.embed_dim, alpha_max, alpha_init, projection_std)
        self.phase = 'B'
        self.udr_stats = {}

    def configure_phase(self, phase):
        if phase not in ('A', 'B'):
            raise ValueError('UDR training phase must be A or B.')
        self.phase = phase
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(phase == 'B' or name.startswith('udr.'))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, 'phase', None) == 'A':
            # Freeze runtime behavior too (e.g. dropout), not just optimizer membership.
            for name, child in self.named_children():
                if name != 'udr':
                    child.eval()
        return self

    def load_rgb_state_dict(self, state):
        """Strict RGB transfer: ONLY new udr.* keys may be absent, never RGB keys."""
        own = self.state_dict()
        rgb_keys = {key for key in own if not key.startswith('udr.')}
        missing, unexpected = rgb_keys - set(state), set(state) - rgb_keys
        if missing or unexpected:
            raise RuntimeError(f'Invalid RGB baseline: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
        for key in rgb_keys:
            if state[key].shape != own[key].shape:
                raise RuntimeError(f'RGB checkpoint shape mismatch: {key}: {state[key].shape} != {own[key].shape}')
        own.update(state)
        self.load_state_dict(own, strict=True)

    @staticmethod
    def _pad_to_size(x, height, width):
        # Same symmetric extension as baseline; repeated extension handles tiny inputs.
        while x.shape[-2] < height:
            x = torch.cat([x, x.flip([2])], 2)
        while x.shape[-1] < width:
            x = torch.cat([x, x.flip([3])], 3)
        return x[..., :height, :width]

    def forward_features(self, x, params):
        size = x.shape[-2:]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        stage_ambiguities = []
        for stage, layer in enumerate(self.layers, start=1):
            # Depth is intentionally absent from ALL ASSB/ASSM parameters.
            stage_params = dict(attn_mask=params['attn_mask'], rpi_sa=params['rpi_sa'])
            collector = []
            if stage >= 4:
                stage_params['ambiguity_collector'] = collector
            x = layer(x, size, stage_params)
            if stage >= 4:
                # Mean over ASSMs within each stage, then equal-weight mean of stages.
                stage_ambiguities.append(torch.stack(collector).mean(0))
        feature = self.patch_unembed(self.norm(x), size)
        ambiguity = torch.stack(stage_ambiguities).mean(0)
        # The only fusion point: after baseline final norm, immediately before conv_after_body.
        feature, self.udr_stats = self.udr(feature, params['rgb'], params['depth'], ambiguity)
        return feature

    def forward(self, rgb, depth):
        if rgb.ndim != 4 or depth.ndim != 4 or rgb.shape[1] != 3 or depth.shape[1] != 1:
            raise ValueError('Expected RGB [B,3,H,W] and full-image-normalized depth [B,1,H,W].')
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError('RGB/depth must be aligned; implicit resizing is forbidden.')
        h0, w0 = rgb.shape[-2:]
        if min(h0, w0) < 1:
            raise ValueError('Empty input is not supported.')
        h = (h0 + self.window_size - 1) // self.window_size * self.window_size
        w = (w0 + self.window_size - 1) // self.window_size * self.window_size
        rgb = self._pad_to_size(rgb, h, w)
        depth = self._pad_to_size(depth, h, w)
        params = dict(attn_mask=self.calculate_mask([h, w]).to(rgb.device),
                      rpi_sa=self.relative_position_index_SA, rgb=rgb, depth=depth)
        mean = self.mean.to(rgb)
        x = self.conv_first((rgb - mean) * self.img_range)
        x = self.conv_after_body(self.forward_features(x, params)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x)) / self.img_range + mean
        return x[..., :h0 * self.upscale, :w0 * self.upscale]
