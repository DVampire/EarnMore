import torch
import torch.nn as nn
import math
from typing import List
from functools import partial
from torch.distributions.normal import Normal
from timm.models.layers import Mlp

from pm.registry import NET
from pm.registry import EMBED

@NET.register_module(force=True)
class ActorTD3(nn.Module):
    def __init__(self,
                 *args,
                 embed_type: str = "TimesEmbed",
                 input_dim = 102,
                 temporal_dim = 3,
                 in_chans: int = 1,
                 embed_dim: int = 128,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = True,
                 explore_noise_std: float = 0.1,
                 **kwargs
                 ):
        super(ActorTD3, self).__init__()
        self.embed_type = embed_type,
        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed
        self.explore_noise_std = explore_noise_std
        ## 这里可能要修改

        self.emb_config = dict(
            type=embed_type,
            input_dim=input_dim,
            temporal_dim=temporal_dim,
            embed_dim=embed_dim,
        )
        self.patch_embed = EMBED.build(self.emb_config)

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Mlp(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.decoder_pred = nn.Linear(
            embed_dim,
            1,
            bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        """
        b, c, n, d, f = x.shape # batch size, in chans, num stocks, days, features
        """
        x = self.patch_embed(x)

        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        x = self.decoder_pred(x)
        return x

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        return pred.tanh()

    def get_action(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        action = pred.tanh()
        noise = (torch.randn_like(action) * self.explore_noise_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)

    def get_action_noise(self, x, action_std):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        action = pred.tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)

@NET.register_module(force=True)
class CriticTD3(nn.Module):
    def __init__(self,
                 *args,
                 embed_type: str = "TimesEmbed",
                 input_dim=102,
                 temporal_dim=3,
                 in_chans: int = 1,
                 embed_dim: int = 128,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = True,
                 **kwargs
                 ):
        super(CriticTD3, self).__init__()

        self.embed_type = embed_type
        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed

        self.emb_config = dict(
            type=embed_type,
            input_dim=input_dim,
            temporal_dim=temporal_dim,
            embed_dim=embed_dim,
        )
        self.patch_embed = EMBED.build(self.emb_config)

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Mlp(in_features=embed_dim + 1, hidden_features=embed_dim + 1, out_features=embed_dim + 1)
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim + 1)

        self.decoder_pred = nn.Linear(
            embed_dim + 1,
            2,
            bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, action):
        """
        b, c, n, d, f = x.shape # batch size, in chans, num stocks, days, features
        """
        x = self.patch_embed(x)

        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if len(action.shape) == 2:
            action = action.unsqueeze(-1)

        x = torch.concat([x, action], dim=-1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        # predictor projection
        x = self.decoder_pred(x)
        return x

    def forward(self, x, action):

        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        latent = self.forward_encoder(x, action)
        pred = self.forward_decoder(latent)

        values = pred.mean(dim=1)

        values = values.mean(dim=1)
        return values

    def get_q_min(self, x, action):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        latent = self.forward_encoder(x, action)
        pred = self.forward_decoder(latent)

        values = pred.mean(dim=1)

        values = values.min(dim=1)[0]
        return values

    def get_q1_q2(self, x, action):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        latent = self.forward_encoder(x, action)
        pred = self.forward_decoder(latent)

        values = pred.mean(dim=1)

        return values[:, 0], values[:, 1]


if __name__ == '__main__':
    device = torch.device("cpu")

    feature = torch.randn(4, 1, 420, 10, 99)
    temporal = torch.zeros(4, 1, 420, 10, 3)

    batch = torch.cat([feature, temporal], dim=-1).to(device)

    model = ActorTD3().to(device)
    pred = model(batch)
    print(pred.shape)

    action = model.get_action(batch)
    print(pred.shape)

    action = model.get_action_noise(batch, 0.1)
    print(action.shape)

    action = torch.randn((4, 421)).to(device)
    model = CriticTD3().to(device)
    values = model(batch, action)
    print(values.shape)

    values = model.get_q_min(batch, action)
    print(values.shape)

    values1, values2 = model.get_q1_q2(batch, action)
    print(values1.shape, values2.shape)