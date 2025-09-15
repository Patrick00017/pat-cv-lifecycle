import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_grad_cam import GradCAM
from models.simple_autoencoder import SimpleAutoEncoder
from models.vision_transformer import ViT
from data.common_datasets import get_mnist_train_loader, get_cifar10_datamodule
from utils.misc import vit_grad_cam_reshape_transform
from pprint import pprint
from tqdm import tqdm
import yaml
import os

seed_everything(7)

# read yaml config file
with open("configs/vit.yaml", "r") as file:
    cfg = yaml.safe_load(file)

# dataloader = get_mnist_train_loader(cfg["train"]["batch_size"])
cifar10_dm = get_cifar10_datamodule(cfg["train"]["batch_size"])

model = ViT(cfg)
model.datamodule = cifar10_dm

trainer = Trainer(
    max_epochs=cfg["train"]["epoch"],
    gpus=0,
    logger=TensorBoardLogger("logs/", name="vit_cifar10"),
    callbacks=[LearningRateMonitor(logging_interval="step")],
)

trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)

# save last ckpt
output_dir = cfg["train"]["output_dir"]
model_name = cfg["model"]["name"]
final_dir_path = f"{output_dir}/{model_name}"
if not os.path.exists(final_dir_path):
    os.makedirs(final_dir_path)
torch.save(model.state_dict(), f"{final_dir_path}/last_checkpoint.pth")
pprint(
    f"train process complete. save checkpoint at {final_dir_path}/last_checkpoint.pth"
)
