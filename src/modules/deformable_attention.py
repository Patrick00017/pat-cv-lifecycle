import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return rearrange(x, "b h w c -> b c h w")


class DAttn(nn.Module):
    def __init__(
        self,
        src_shape: tuple,  # should be (C, H, W)
        k: int,
        r: int,
    ):
        super().__init__()
        self.H, self.W, self.C = src_shape

        self.theta_offset = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=k, stride=r),
            nn.GELU(),
            nn.Conv2d(self.C, 2, kernel_size=1, stride=1),
        )

        self.proj_q = nn.Conv2d(self.C, self.C, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.C, self.C, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.C, self.C, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.C, self.C, kernel_size=1, stride=1, padding=0)

    @torch.no_grad()
    def get_reference_points(self, B, h_r, w_r, dtype, device):
        """
        This function is used to generate a matrix of reference points.
        Size: (B, h/r, w/r)
        Top-left is (-1, -1) and right-bottom is (1, 1)
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, h_r - 0.5, h_r, dtype=dtype, device=device),
            torch.linspace(0.5, w_r - 0.5, w_r, dtype=dtype, device=device),
            indexing="ij",
        )
        ref = torch.stack((ref_y, ref_x), -1)  # merge to (h_r, w_r, 2)
        ref[..., 1].div_(w_r - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(h_r - 1.0).mul_(2.0).sub_(1.0)
        ref = repeat(ref, "h w p -> B h w p", B=B)
        return ref

    def forward(self, x):
        B = x.shape[0]
        q = self.proj_q(x)  # q -> (B, c, h, w)
        offsets = self.theta_offset(q)  # offsets -> (B, 2, h_r, w_r)
        h_r, w_r = offsets.shape[2:]
        n_sample = h_r * w_r
        ref_points = self.get_reference_points(B, h_r, w_r, x.dtype, x.device)
        print(ref_points.shape, offsets.shape)

        offsets = rearrange(offsets, "b p h w -> b h w p")
        deformed_points = (ref_points + offsets).clamp(-1.0, +1.0)
        x_tilde = F.grid_sample(
            input=x,
            grid=deformed_points[..., (1, 0)],  # (y, x) -> (x, y)
            mode="bilinear",
            align_corners=True,
        )  # (B, c, h_r, w_r)
        x_tilde = rearrange(x_tilde, "b c hr wr -> b c 1 (hr wr)")

        q = rearrange(q, "b c h w -> b c (h w)")
        k = self.proj_k(x_tilde)
        k = rearrange(k, "b c 1 n -> b c n")
        v = self.proj_v(x_tilde)
        v = rearrange(v, "b c 1 n -> b c n")

        attn = torch.einsum("b c m, b c n -> b m n", q, k)  # B * h, HW, Ns
        # attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        # attn = self.attn_drop(attn)
        out = torch.einsum("b m n, b c n -> b c m", attn, v)
        out = rearrange(out, "b c (h w) -> b c h w", h=self.H, w=self.W)
        # y = self.proj_drop(self.proj_out(out))
        return out


def build_deformable_attention(input_shape: tuple, k: int, r: int):
    """
    This function is used to build a deformable attention module.

    params:
        input_shape: (c, h, w)
        k: kernel size
        r: stride size
    """
    module = DAttn(input_shape, k, r)
    return module


if __name__ == "__main__":
    dattn = build_deformable_attention((100, 100, 24), 3, 1)
    print(dattn.get_reference_points(10, 100, 100, torch.float32, "cpu").shape)

    x = torch.rand((10, 24, 100, 100), dtype=torch.float32)
    out = dattn(x)
    print(out.shape)
