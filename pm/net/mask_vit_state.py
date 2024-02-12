import torch
import torch.nn as nn
from typing import List
from functools import partial
from pm.registry import NET
from pm.net import MAE

@NET.register_module(force=True)
class MaskVitState(MAE):
    def __init__(self,
                *args,
                embed_type: str = "PatchEmbed",
                feature_size: List[int] = (10, 99),
                patch_size: List[int] = (10, 99),
                t_patch_size: int = 1,
                num_stocks: int = 420,
                pred_num_stocks: int = 420,
                in_chans: int = 1,
                embed_dim: int = 128,
                depth: int = 2,
                num_heads: int = 4,
                decoder_embed_dim: int = 64,
                decoder_depth: int = 1,
                decoder_num_heads: int = 8,
                mlp_ratio: float = 4.0,
                norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                norm_pix_loss: bool = False,
                cls_embed: bool = True,
                sep_pos_embed: bool = True,
                trunc_init: bool = False,
                no_qkv_bias: bool = False,
                mask_ratio_min: float = 0.5,
                mask_ratio_max: float = 1.0,
                mask_ratio_mu: float = 0.55,
                mask_ratio_std:float = 0.25,
                ** kwargs
                ):
        super(MaskVitState, self).__init__(
            *args,
            embed_type=embed_type,
            feature_size = feature_size,
            patch_size = patch_size,
            t_patch_size = t_patch_size,
            num_stocks = num_stocks,
            pred_num_stocks = pred_num_stocks,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            decoder_embed_dim = decoder_embed_dim,
            decoder_depth = decoder_depth,
            decoder_num_heads = decoder_num_heads,
            mlp_ratio = mlp_ratio,
            norm_layer = norm_layer,
            norm_pix_loss = norm_pix_loss,
            cls_embed = cls_embed,
            sep_pos_embed = sep_pos_embed,
            trunc_init = trunc_init,
            no_qkv_bias = no_qkv_bias,
            mask_ratio_min = mask_ratio_min,
            mask_ratio_max = mask_ratio_max,
            mask_ratio_mu = mask_ratio_mu,
            mask_ratio_std = mask_ratio_std,
            **kwargs,
        )

        self.initialize_weights()

    def forward_encoder(self, x, mask = None, ids_restore = None):
        """
        b, c, n, d, f = x.shape # batch size, in chans, num stocks, days, features
        """

        x = self.patch_embed(x)

        B, L, C = x.shape

        if mask is None:
            mask_ratio = self.mask_ratio_generator.rvs(1)[0]
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
            x = x.view(B, -1, C)
        else:
            ids_keep = torch.argsort(ids_restore, dim=1)[:, :(mask[0, :] == 0).sum().item()]
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1],
                dim=1,
            )

            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )

        x = x.view([B, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])

        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def forward_state(self,x, mask = None, ids_restore = None):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        x, mask, ids_restore = self.forward_encoder(x,
                                                    mask = mask,
                                                    ids_restore = ids_restore)

        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])

        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        _imgs = torch.index_select(
            imgs,
            2,
            torch.linspace(
                0,
                imgs.shape[2] - 1,
                self.num_stocks,
            )
            .long()
            .to(imgs.device),
        )
        target = self.patchify(_imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, mask = None, ids_restore = None):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        latent, mask, ids_restore = self.forward_encoder(x,
                                                         mask = mask,
                                                         ids_restore = ids_restore)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, mask, ids_restore

if __name__ == '__main__':
    device = torch.device("cpu")

    model = MaskVitState().to(device)
    print(model)

    batch = torch.randn((4, 1, 420, 10, 99)).to(device)
    loss, mask, ids_restore = model(batch)
    print(loss, mask.shape, ids_restore.shape)