import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import pytorch_lightning as pl
from torchmetrics.functional import accuracy


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, drop_prob=0.1):
        super().__init__()

        self.patch_size = patch_size
        self.in_feats = (patch_size**2) * 3  # 3 is image channel
        self.layers = nn.Sequential(
            nn.LayerNorm(self.in_feats),
            nn.Linear(self.in_feats, embed_dim),
            nn.Dropout(drop_prob),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        x = rearrange(x, "b c (h p) (w p) -> b (h w) (c p p)", p=self.patch_size)
        return self.layers(x)


class PositionEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=2**12):
        super().__init__()

        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(embed_dim // 2).unsqueeze(0)
        angle = pos / (10_000 ** (2 * i / embed_dim))
        self.pe_mat = torch.zeros(size=(max_len, embed_dim))
        self.pe_mat[:, 0::2] = torch.sin(angle)
        self.pe_mat[:, 1::2] = torch.cos(angle)
        self.register_buffer("pos_enc_mat", self.pe_mat)


class PatchShuffle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask_ratio):
        n_tokens = x.size(1)
        # generate random shuffle indices
        shuffle_indices = torch.randperm(n_tokens)
        # get the last mask ratio indices
        mask_indices = shuffle_indices[-int(n_tokens * mask_ratio) :]
        mask = torch.isin(torch.arange(n_tokens), mask_indices)
        return x[:, ~mask, :], mask


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dims, n_heads, mlp_ratio=4):
        super(TransformerBlock, self).__init__()
        self.hidden_dims = hidden_dims
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_dims)
        self.mhsa = nn.MultiheadAttention(hidden_dims, n_heads)
        self.norm2 = nn.LayerNorm(hidden_dims)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims, mlp_ratio * hidden_dims),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_dims, hidden_dims),
        )

    def forward(self, x):
        x = self.norm1(x)
        attn_output, _ = self.mhsa(query=x, key=x, value=x)
        out = x + attn_output
        out = out + self.mlp(self.norm2(out))
        return out


class Encoder(nn.Module):
    def __init__(self, chw, patch_size, depth, embed_dim, n_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.cell_size = chw[1:] // patch_size  # (num_patch ** 2)
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        self.pos_embed = PositionEmbedding(embed_dim)
        self.patch_shuffle = PatchShuffle()
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, n_heads) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mask_token = nn.Parameter(torch.randn((embed_dim,)))

    def forward(self, image, mask_ratio):
        x = self.patch_embed(image)
        x += repeat(self.pos_embed.pe_mat.to(x.device), "l d -> b l d", b=x.size(0))[
            :, : x.size(1), :
        ]
        unmasked_token, mask = self.patch_shuffle(x, mask_ratio=mask_ratio)
        x = self.blocks(unmasked_token)
        x = self.norm(x)

        new_token = repeat(
            self.mask_token, "w -> b c w", b=x.size(0), c=self.cell_size**2
        ).clone()
        new_token[:, ~mask, :] = x
        return new_token, mask

class Decoder(nn.Module):
    def __init__(self, chw, patch_size, depth, embed_dim, n_heads, n_pixel_values=256):
        super().__init__()
        self.chw = chw
        self.patch_size = patch_size
        self.cell_size = chw[1:] // patch_size

        self.pos_embed = PositionEmbedding(embed_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, n_heads) for _ in range(depth)
        )
        self.proj = nn.Linear(embed_dim, (patch_size ** 2) * 3)

    def forward(self, x):
        # x shape is [b, cells, embed_dim]
        x += repeat(self.pos_embed.pe_mat.to(x.device), "l d -> b l d", b=x.size(0))[:, :x.size(1), :]
        x = self.blocks(x)
        x = self.proj(x)
        x = rearrange(x, "b (h w) (p p 3) -> b 3 (h p) (w p)", h=self.chw[1], w=self.chw[2], p=self.patch_size)
        return x
    
class MAE(pl.LightningModule):
    def __init__(self, chw, patch_size, n_heads, encoder_depth, decoder_depth, encoder_embed_dim, decoder_embed_dim):
        super().__init__()
        self.chw = chw
        self.patch_size = patch_size
        assert chw[1] % patch_size == 0 and chw[2] % patch_size == 0, "img size must be divided by patch size"
        self.n_cells = chw[1] // patch_size
        self.encoder = Encoder(chw, patch_size, encoder_depth, encoder_embed_dim, n_heads)
        self.proj = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.decoder = Decoder(chw, patch_size, decoder_depth, decoder_embed_dim, n_heads)

    def forward(self, x, mask_ratio=0.75):
        x, mask = self.encoder(x, mask_ratio)
        x = self.proj(x)
        x = self.decoder(x)
        return x, mask

    def upsample_mask(self, mask, batch_size, device):
        up_mask = torch.repeat_interleave(
            torch.repeat_interleave(
                rearrange(mask, '(c c) -> c c'),
                repeats=self.patch_size,
                dim=0
            ),
            repeats=self.patch_size,
            dim=1
        ).to(device) # (p p c c)
        up_mask = repeat(up_mask, 'h w -> b c h w', b=batch_size, c=3)
        return up_mask

    def get_loss(self, image, mask_ratio=0.75):
        out, mask = self.forward(image, mask_ratio)
        up_mask = self.upsample_mask(mask, image.size(0), image.device)
        loss = up_mask * F.mse_loss(out, image, reduction='none')
        return torch.mean(loss)

    def reconstruct(self, image, mask_ratio=0.75):
        out, mask = self.forward(image, mask_ratio)
        up_mask = self.upsample_mask(mask, image.size(0), image.device)
        masked_image = torch.where(up_mask, torch.full_like(image, fill_value=0), image)
        recon_image = torch.where(up_mask, out, image)
        return masked_image, recon_image
        