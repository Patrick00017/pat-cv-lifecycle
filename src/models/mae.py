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
