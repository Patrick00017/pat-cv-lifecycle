import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import OneCycleLR


def patchify(images, n_patches):
    n, c, h, w = images.shape
    # check images h and w is the same
    assert h == w, "patchify method is implemented for square images only"

    # patch and patch values
    patches = torch.zeros(n, n_patches**2, h * w * c // n_patches**2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


def unpatchify(patches, img_size):
    n, n_patches, chw = patches.shape
    features = rearrange(patches, "b n v -> b (n v)")
    print(features.shape)
    images = rearrange(features, "b (c h w) -> b c h w", h=img_size[0], w=img_size[1])
    # assert images.shape[1] == 3, "imgs channel should be 3"
    return images


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


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


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_patches,
        patch_size,
        embed_dims,
        decoder_embed_dims,
        n_heads,
        chw,
        mlp_ratio=4,
        depth=4,
    ):
        super(TransformerDecoder, self).__init__()
        c, h, w = chw

        # mapper latent space features to decoder dims
        self.decoder_embed_mapper = nn.Linear(embed_dims, decoder_embed_dims)
        # learnable position encoding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, n_patches**2, decoder_embed_dims), requires_grad=True
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dims=decoder_embed_dims, n_heads=n_heads)
                for _ in range(depth)
            ]
        )
        self.mlp = nn.Sequential(
            nn.Linear(decoder_embed_dims, decoder_embed_dims * mlp_ratio),
            nn.GELU(),
            nn.Linear(decoder_embed_dims * mlp_ratio, int(patch_size[0]) ** 2 * c),
        )

    def forward(self, x):
        b = x.shape[0]
        x = self.decoder_embed_mapper(x)
        # refined_pos_embed = repeat(self.decoder_pos_embed, 'n d -> b n d', b=b)
        x += self.decoder_pos_embed
        for block in self.blocks:
            x = block(x)
        return x


class ViT(pl.LightningModule):
    def __init__(self, cfg):
        super(ViT, self).__init__()
        # cfg is model config settings
        self.cfg = cfg
        self.chw = cfg["model"]["chw"]
        self.n_patches = cfg["model"]["num_patches"]
        assert (
            self.chw[1] % self.n_patches == 0
        ), "Input shape not entirely divisible by patchsize"
        assert (
            self.chw[2] % self.n_patches == 0
        ), "Input shape not entirely divisible by patchsize"

        self.patch_size = (self.chw[1] / self.n_patches, self.chw[2] / self.n_patches)
        self.hidden_dims = cfg["model"]["hidden_dims"]
        # 1. linear mapper
        self.input_dim = int(self.chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_dim, self.hidden_dims)
        # 2. cls token
        self.cls_token = nn.Parameter(torch.rand(1, self.hidden_dims))
        # 3. pos embed
        self.pos_embed = nn.Parameter(
            torch.tensor(
                get_positional_embeddings(self.n_patches**2 + 1, self.hidden_dims)
            )
        )
        self.pos_embed.requires_grad = False
        # 4. transformer encoder
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dims=self.hidden_dims, n_heads=cfg["model"]["n_heads"]
                )
                for _ in range(cfg["model"]["n_encoder_blocks"])
            ]
        )
        # 5. classification mlp
        self.mlp = nn.Linear(self.hidden_dims, cfg["dataset"]["num_classes"])

    def forward(self, images):
        batch_size = images.shape[0]
        patches = patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)
        # merge cls token
        print(self.cls_token.shape)
        refined_cls_token = repeat(self.cls_token, "n d -> b n d", b=batch_size)
        tokens = torch.cat([refined_cls_token, tokens], dim=1)
        refined_pos_embed = rearrange(self.pos_embed, "n d -> 1 n d")
        out = tokens + refined_pos_embed
        for block in self.encoder:
            out = block(out)
        # use cls token only
        out = self.mlp(out[:, 0])
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["train"]["lr"])
        return optimizer

    def feature_heatmap_target_layer(self):
        if self.cfg["model"]["visualization"]["enable"]:
            return self.encoder[-1].norm1
        return None


class ViT_Autoencoder(nn.Module):
    def __init__(
        self,
        chw=(1, 28, 28),
        n_patches=7,
        hidden_dims=128,
        n_heads=8,
        n_blocks=4,
        decoder_hidden_dims=128,
    ):
        super(ViT_Autoencoder, self).__init__()

        self.chw = chw
        self.n_patches = n_patches
        assert (
            chw[1] % n_patches == 0
        ), "Input shape not entirely divisible by patchsize"
        assert (
            chw[2] % n_patches == 0
        ), "Input shape not entirely divisible by patchsize"

        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
        self.hidden_dims = hidden_dims
        # 1. linear mapper
        self.input_dim = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_dim, self.hidden_dims)
        # 2. cls token
        self.cls_token = nn.Parameter(torch.rand(1, self.hidden_dims))
        # 3. pos embed
        self.pos_embed = nn.Parameter(
            torch.tensor(
                get_positional_embeddings(self.n_patches**2 + 1, self.hidden_dims)
            )
        )
        self.pos_embed.requires_grad = False
        # 4. transformer encoder
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(hidden_dims=hidden_dims, n_heads=n_heads)
                for _ in range(n_blocks)
            ]
        )

        self.decoder = TransformerDecoder(
            n_patches=n_patches,
            patch_size=self.patch_size,
            embed_dims=hidden_dims,
            decoder_embed_dims=decoder_hidden_dims,
            n_heads=n_heads,
            chw=chw,
        )

    def forward(self, images):
        batch_size = images.shape[0]
        patches = patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)
        # merge cls token
        # print(self.cls_token.shape)
        refined_cls_token = repeat(self.cls_token, "n d -> b n d", b=batch_size)
        tokens = torch.cat([refined_cls_token, tokens], dim=1)
        refined_pos_embed = rearrange(self.pos_embed, "n d -> 1 n d")
        out = tokens + refined_pos_embed
        for block in self.encoder:
            out = block(out)
        # use cls token only
        out = self.decoder(out[:, 1:])
        return out


# img_size = (28, 28)
# images = torch.randn((12, 1, 28, 28))
# # model = ViT()
# model = ViT_Autoencoder()
# out = model(images)
# out = unpatchify(out, (28, 28))
# print(out.shape)
# model = TransformerEncoderBlock(hidden_dims=8, n_heads=2)
# x = torch.randn(7, 50, 8)  # Dummy sequences
# print(model(x).shape)      # torch.Size([7, 50, 8])
