import os
import torch
from torch import nn
import torch.nn.functional as F
import einops
import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(cfg["model"]["img_width"] * cfg["model"]["img_height"], cfg["model"]["encoder"]["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["model"]["encoder"]["hidden_dim"], cfg["model"]["latent_dim"])
        )
    def forward(self, x):
        return self.l1(x)
    
class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(cfg["model"]["latent_dim"], cfg["model"]["decoder"]["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["model"]["decoder"]["hidden_dim"], cfg["model"]["img_width"] * cfg["model"]["img_height"])
        )
    def forward(self, x):
        return self.l1(x)
    
class SimpleAutoEncoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = einops.rearrange(x, 'b c h w -> b c (h w)')
        # x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["train"]["lr"])
        return optimizer