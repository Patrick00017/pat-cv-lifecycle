import os
import torch
from torch import nn
import torch.nn.functional as F
import einops
import pytorch_lightning as pl
from data.self_driving_simulator_dataset import SelfDrivingDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pprint import pprint
import matplotlib.pyplot as plt


class SelfDrivingCNNModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
        )
        self.dropout = nn.Dropout(p=0.5)
        self.head = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = einops.rearrange(x, "b c h w -> b (c h w)")
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, steering = batch  # x: [batchsize, 3, 66, 200]
        yhat = self.forward(x)
        loss = F.mse_loss(yhat, steering.float())
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer


if __name__ == "__main__":
    pl.seed_everything(7)

    dm = SelfDrivingDataModule(batch_size=16)

    model = SelfDrivingCNNModel()
    # model = model.to("cuda")

    trainer = pl.Trainer(
        max_epochs=30,
        # accelerator="gpu",
        # devices=[0],
        # logger=TensorBoardLogger("logs/", name="self-driving"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    # save last ckpt
    output_dir = "./self-driving"
    model_name = "self-driving-model"
    final_dir_path = f"{output_dir}/{model_name}"
    if not os.path.exists(final_dir_path):
        os.makedirs(final_dir_path)
    torch.save(model.state_dict(), f"{final_dir_path}/last_checkpoint.pth")
    pprint(
        f"train process complete. save checkpoint at {final_dir_path}/last_checkpoint.pth"
    )
