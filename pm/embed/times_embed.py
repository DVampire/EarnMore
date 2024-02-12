import torch
import math
import torch.nn as nn
from pm.registry import EMBED
from einops import rearrange
from typing import List
from timm.models.layers import to_2tuple

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float()
                    * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels = input_dim,
                                   out_channels=embed_dim,
                                   kernel_size=3,
                                   padding=padding,
                                   padding_mode='circular',
                                   bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(input_dim, embed_dim).float()
        w.require_grad = False

        position = torch.arange(0, input_dim).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float()
                    * -(math.log(10000.0) / embed_dim)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(input_dim, embed_dim)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, embed_dim, embed_type='timeF'):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(3, embed_dim, bias=False)
    def forward(self, x):
        return self.embed(x)

class TemporalEmbedding(nn.Module):
    def __init__(self, embed_dim, embed_type='fixed'):
        super(TemporalEmbedding, self).__init__()

        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        self.weekday_embed = Embed(weekday_size, embed_dim)
        self.day_embed = Embed(day_size, embed_dim)
        self.month_embed = Embed(month_size, embed_dim)

    def forward(self, x):
        x = x.long()

        weekday_x = self.weekday_embed(x[:, :, 0])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 2])

        return weekday_x + day_x + month_x

@EMBED.register_module(force=True)
class TimesEmbed(nn.Module):
    def __init__(self,
                 *args,
                 img_size=(10, 102),
                 patch_size=(10, 102),
                 frames=420,
                 t_patch_size=1,
                 input_dim: int = 102,
                 temporal_dim: int = 3,
                 embed_dim: int = 128,
                 embed_type='fixed',
                 **kwargs
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.embed_dim = embed_dim

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = (
                (img_size[1] // patch_size[1])
                * (img_size[0] // patch_size[0])
                * (frames // t_patch_size)
        )
        self.input_size = (
            frames // t_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]
        self.t_grid_size = frames // t_patch_size

        self.feature_dim = self.input_dim - temporal_dim

        self.value_embedding = TokenEmbedding(input_dim=self.feature_dim,
                                              embed_dim=embed_dim)
        self.position_embedding = PositionalEmbedding(embed_dim=embed_dim)
        self.temporal_embedding = TemporalEmbedding(embed_dim=embed_dim,
                                                    embed_type=embed_type) if embed_type != 'timeF' else \
                                  TimeFeatureEmbedding(embed_dim = embed_dim, embed_type=embed_type)

        #self.proj = nn.Linear(self.img_size[0] * embed_dim, embed_dim)


    def forward(self, x):

        B, C, N, D, F = x.shape

        x = rearrange(x, "b c n d f -> (b c n) d f", b = B, c = C, n = N)

        feature = x[..., :-self.temporal_dim]
        temporal = x[..., -self.temporal_dim:]

        x = self.value_embedding(feature) + self.temporal_embedding(temporal) + self.position_embedding(feature)

        x = rearrange(x, "(b c n) d f -> b c n d f", b=B, c=C, n=N)
        x = x.mean(dim=-2).squeeze(1)

        # x = rearrange(x, "(b c n) d f -> (b c n) (d f)", b=B, c=C, n=N)
        # x = self.proj(x)
        # x = rearrange(x, "(b c n) f -> b c n f", b=B, c=C, n=N).squeeze(1)

        return x

@EMBED.register_module(force=True)
class TimesEmbedWoPos(nn.Module):
    def __init__(self,
                 *args,
                 img_size=(10, 102),
                 patch_size=(10, 102),
                 frames = 420,
                 t_patch_size = 1,
                 input_dim: int = 102,
                 temporal_dim: int = 3,
                 embed_dim: int = 128,
                 embed_type='fixed',
                 **kwargs
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.embed_dim = embed_dim

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = (
                (img_size[1] // patch_size[1])
                * (img_size[0] // patch_size[0])
                * (frames // t_patch_size)
        )
        self.input_size = (
            frames // t_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]
        self.t_grid_size = frames // t_patch_size

        self.feature_dim = self.input_dim - temporal_dim

        self.value_embedding = TokenEmbedding(input_dim=self.feature_dim,
                                              embed_dim=embed_dim)
        self.temporal_embedding = TemporalEmbedding(embed_dim=embed_dim,
                                                    embed_type=embed_type) if embed_type != 'timeF' else \
                                  TimeFeatureEmbedding(embed_dim = embed_dim, embed_type=embed_type)

        #self.proj = nn.Linear(self.img_size[0] * embed_dim, embed_dim)

    def forward(self, x):

        B, C, N, D, F = x.shape

        x = rearrange(x, "b c n d f -> (b c n) d f", b = B, c = C, n = N)

        feature = x[..., :-self.temporal_dim]
        temporal = x[..., -self.temporal_dim:]

        x = self.value_embedding(feature) + self.temporal_embedding(temporal)

        x = rearrange(x, "(b c n) d f -> b c n d f", b=B, c=C, n=N)
        x = x.mean(dim=-2).squeeze(1)

        # x = rearrange(x, "(b c n) d f -> (b c n) (d f)", b=B, c=C, n=N)
        # x = self.proj(x)
        # x = rearrange(x, "(b c n) f -> b c n f", b=B, c=C, n=N).squeeze(1)

        return x


if __name__ == '__main__':
    feature = torch.randn(4, 1, 420, 30, 99)
    temporal = torch.zeros(4, 1, 420, 30, 3)

    batch = torch.cat([feature, temporal], dim=-1)

    model = TimesEmbed(
        img_size=(30, 102),
        patch_size=(30, 102),
        frames = 420,
        t_patch_size = 1,
        input_dim = 102,
        temporal_dim= 3,
        embed_dim= 128
    )

    res = model(batch)
    print(res.shape)

    model = TimesEmbedWoPos(
        img_size=(30, 102),
        patch_size=(30, 102),
        frames=420,
        t_patch_size=1,
        input_dim = 102,
        temporal_dim= 3,
        embed_dim= 128
    )
    res = model(batch)
    print(res.shape)