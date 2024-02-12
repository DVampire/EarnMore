import torch
import torch.nn as nn
from pm.registry import EMBED
from timm.models.layers import to_2tuple
from typing import List

@EMBED.register_module(force=True)
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        *args,
        img_size: List[int] = None,
        patch_size: List[int] = None,
        in_chans=3,
        embed_dim=768,
        frames=32,
        t_patch_size=4,
        **kwargs
    ):
        super().__init__()
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

        self.frames = frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]
        self.t_grid_size = frames // t_patch_size

        kernel_size = [t_patch_size] + list(patch_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert T == self.frames
        x = self.proj(x).flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]

        B, T, L, C = x.shape
        x = x.reshape(B, T * L, C)

        return x


if __name__ == '__main__':
    device = torch.device("cpu")

    model = PatchEmbed(img_size=(10, 99),
                       patch_size=(10, 99),
                       in_chans=1,
                       embed_dim=128,
                       frames=420,
                       t_patch_size=1).to(device)
    print(model)
    print(model.num_patches)
    print(model.input_size)

    batch = torch.randn((4, 1, 420, 10, 99)).to(device)
    emb = model(batch)
    print(emb.shape)