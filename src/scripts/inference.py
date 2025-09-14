import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from models.simple_autoencoder import SimpleAutoEncoder
from models.vision_transformer import ViT
from data.common_datasets import get_mnist_train_loader, get_cifar10_datamodule
from pprint import pprint
from tqdm import tqdm
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import yaml
import os

seed_everything(7)

# read yaml config file
with open("configs/vit.yaml", "r") as file:
    cfg = yaml.safe_load(file)

targets = [ClassifierOutputSoftmaxTarget(281)]
model = ViT(cfg)
# if model config contains attention visualze model like grad-cam, so init it
cam = None
if cfg["model"]["visualization"]["enable"]:
    cam = GradCAM(model=model, target_layers=model.get_feature_heatmap_target_layer(), reshape_transform=vit_grad_cam_reshape_transform)

# todo: read image
input_tensor = None
rgb_img = None

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
# You can also get the model outputs without having to redo inference
model_outputs = cam.outputs
