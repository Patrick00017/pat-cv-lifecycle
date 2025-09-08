import os
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

class Encoder(nn.Module):
    def __init__(self):
        super.__init__()
        self.l1 = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.l1(x)
    
class Decoder(nn.Module):
    def __init__(self):
        super.__init__()
        self.l1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28)
        )
    def forward(self, x):
        return self.l1(x)
    
class SimpleAutoEncoder(L.LightningModule):
    def __init__(self):
        super.__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer