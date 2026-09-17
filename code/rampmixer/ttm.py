"""Ramp-oriented temporal mixing and contribution-aligned context attention."""
import torch
from torch import nn


class TemporalMixer(nn.Module):
    def __init__(self, patches, hidden, expansion=2):
        super().__init__()
        self.ln_time, self.ln_feature = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.inter = nn.Sequential(nn.Linear(patches, patches*expansion), nn.GELU(),
                                   nn.Linear(patches*expansion, patches))
        self.intra = nn.Sequential(nn.Linear(hidden, hidden*expansion), nn.GELU(),
                                   nn.Linear(hidden*expansion, hidden))

    def forward(self, x):
        x = x+self.inter(self.ln_time(x).transpose(-1, -2)).transpose(-1, -2)
        return x+self.intra(self.ln_feature(x))


class RampMixer(nn.Module):
    """Section 3.3 patch/descriptor -> independent mixer -> context attention.

    Trained from scratch; not an IBM pretrained TTM checkpoint. Parameter sharing
    across modes never mixes their activations inside the temporal encoder.
    """
    def __init__(self, modes, length=96, horizon=12, patch=6, hidden=64,
                 layers=2, heads=4, head_hidden=128, mode_encoding=True):
        super().__init__()
        if length % patch:
            raise ValueError('Historical length must be divisible by patch length')
        self.patch, self.mode_encoding = patch, mode_encoding
        patches = length//patch
        self.patch_embed = nn.Linear(patch, hidden)
        self.driver_identity = nn.Embedding(modes, hidden)
        self.desc_embed = nn.Linear(2, hidden)
        self.encoder = nn.ModuleList([TemporalMixer(patches, hidden) for _ in range(layers)])
        self.context_embed = nn.Linear(patch, hidden)
        self.context_identity = nn.Embedding(3, hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.prediction = nn.Sequential(nn.Linear(patches*hidden, head_hidden), nn.GELU(),
                                        nn.Linear(head_hidden, horizon))

    def encode(self, modes, descriptors):
        # [batch, time, modes] -> [batch, modes, patches, hidden]
        x = self.patch_embed(modes.transpose(1, 2).unfold(-1, self.patch, self.patch))
        if self.mode_encoding:
            identity = self.driver_identity(torch.arange(x.shape[1], device=x.device))
            x = x+(self.desc_embed(descriptors)+identity)[..., None, :]
        for layer in self.encoder:
            x = layer(x)
        return x

    def forward(self, modes, descriptors, power_context, return_attention=False):
        x = self.encode(modes, descriptors)
        b, m, n, hidden = x.shape
        ctx = self.context_embed(power_context.transpose(1, 2).unfold(-1, self.patch, self.patch))
        ctx = ctx+self.context_identity(torch.arange(3, device=x.device))[None, :, None, :]
        ctx = ctx.reshape(b, 3*n, hidden)
        out, attention = self.attention(x.reshape(b, m*n, hidden), ctx, ctx,
                                        need_weights=return_attention)
        components = self.prediction(out.reshape(b, m, n*hidden))
        prediction = components.sum(1)
        return (prediction, attention) if return_attention else prediction
