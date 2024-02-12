import torch
import torch.nn as nn
from typing import List
from functools import partial
from torch.distributions.normal import Normal
from timm.models.layers import Mlp
from torch.nn import functional as F
from pm.registry import NET
from pm.net import MaskTimeState

@NET.register_module(force=True)
class ActorMaskSAC(nn.Module):
    def __init__(self,
                *args,
                embed_dim: int = 128,
                depth: int = 2,
                norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                cls_embed: bool = True,
                ** kwargs
                ):
        super(ActorMaskSAC, self).__init__()
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed

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
            2,
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
            torch.nn.init.orthogonal_(m.weight, 1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        """
        b, l, c = x.shape # batch size, num stocks + 1, embed_dim
        """
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        x = self.decoder_pred(x)
        return x

    def forward(self, x):
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)

        logits = pred[:, :, 0]

        indices = torch.sort(logits)[1]
        soft_logits = logits * torch.log(indices + 1)

        weight = F.softmax(soft_logits, dim=-1).squeeze(-1)

        return weight

    def get_action(self, x):

        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)

        a_avg, a_std_log = pred.chunk(2, dim=-1)
        a_std = a_std_log.clamp(-16, 2).exp()
        a_avg = a_avg.squeeze(-1)
        a_std = a_std.squeeze(-1)

        dist = Normal(a_avg, a_std)

        logits = dist.rsample()
        indices = torch.sort(logits)[1]
        soft_logits = logits * torch.log(indices + 1)

        weight = F.softmax(soft_logits, dim=-1).squeeze(-1)

        return weight

    def get_action_logprob(self, x):

        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)

        a_avg, a_std_log = pred.chunk(2, dim=-1)
        a_std = a_std_log.clamp(-16, 2).exp()
        a_avg = a_avg.squeeze(-1)
        a_std = a_std.squeeze(-1)

        dist = Normal(a_avg, a_std)
        logits = dist.rsample()

        indices = torch.sort(logits)[1]
        soft_logits = logits * torch.log(indices + 1)

        weight = F.softmax(soft_logits, dim=-1).squeeze(-1)

        logprob = dist.log_prob(a_avg)
        logprob -= (-weight.pow(2) + 1.000001).log()

        return weight, logprob.sum(1)


@NET.register_module(force=True)
class CriticMaskSAC(nn.Module):
    def __init__(self,
                 *args,
                 embed_dim: int = 128,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = True,
                 **kwargs
                 ):
        super(CriticMaskSAC, self).__init__()

        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed

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

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, 1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_encoder(self, x, action):
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if len(action.shape) == 2:
            action = action.unsqueeze(-1)
        x = torch.concat([x, action], dim=-1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        x = self.decoder_pred(x)
        return x

    def forward(self, x, action):

        latent = self.forward_encoder(x, action)
        latent = torch.sum(latent, dim=1)
        pred = self.forward_decoder(latent)
        value = pred.mean(dim=1)
        return value

    def get_q_min(self, x, action):

        latent = self.forward_encoder(x, action)
        latent = torch.sum(latent, dim=1)
        pred = self.forward_decoder(latent)
        value = pred.min(dim=1)[0]
        return value

    def get_q1_q2(self, x, action):

        latent = self.forward_encoder(x, action)
        latent = torch.sum(latent, dim=1)
        pred = self.forward_decoder(latent)
        value = pred

        return value[:, 0], value[:, 1]


if __name__ == '__main__':
    device = torch.device("cpu")

    feature = torch.randn(4 * 4, 420, 10, 99)
    temporal = torch.zeros(4 * 4, 420, 10, 3)

    batch = torch.cat([feature, temporal], dim=-1).to(device)

    model = MaskTimeState(
        feature_size=(10, 102),
        patch_size=(10, 102),
        frames=420,
        t_patch_size=1,
        input_dim=102,
        temporal_dim=3,
        embed_dim=128
    ).to(device)
    loss, mask, ids_restore = model(batch)
    print(loss, mask.shape, ids_restore.shape)

    state, mask, ids_restore = model.forward_state(batch, mask, ids_restore)
    print(state.shape)

    model = ActorMaskSAC(embed_dim=64).to(device)
    pred = model(state)
    print(pred.shape)

    action = model.get_action(state)
    print(action.shape)

    action, action_logprob = model.get_action_logprob(state)
    print(action.shape, action_logprob.shape)

    action = torch.randn((4*4, 421)).to(device)
    model = CriticMaskSAC(embed_dim=64).to(device)
    values = model(state, action)
    print(values.shape)

    values = model.get_q_min(state, action)
    print(values.shape)

    values1, values2 = model.get_q1_q2(state, action)
    print(values1.shape, values2.shape)