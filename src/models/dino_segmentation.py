import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov2.models.vision_transformer import vit_small
from dinov2.data.transforms import make_classification_eval_transform
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from einops import rearrange
from data.image_segmentation_dataset import SegmentationDataModule
from pytorch_lightning import Trainer

BATCH_SIZE = 36
lr = 0.01
PRECISION="16-mixed"
ID2LABEL={
    0: 'Background',
    1: 'Crack',
}
CHECKPOINT_CALLBACK = ModelCheckpoint(save_top_k=3, 
                                      monitor="valLoss", 
                                      every_n_epochs=1,  # Save the model at every epoch 
                                      save_on_train_epoch_end=True  # Ensure saving happens at the end of a training epoch
                                     )
LOGGER = CSVLogger("outputs", name="lightning_logs_csv")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        # 多分类Dice Loss
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        probs = F.softmax(logits, dim=1)
        
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, weight=None):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss()
    
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss

class DinoSegmentation(pl.LightningModule):
    def __init__(self, chw, num_classes):
        super().__init__()
        self.chw = chw # original channel height width
        # init dinov2 backbone
        self.backbone = torch.hub.load('/mnt/data/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vits14', source='local', pretrained=True)
        # self.backbone = self.backbone.to(device)
        self.backbone.eval()
        self.embed_dim = 384
        self.num_classes = num_classes
        # linear decoder as seg head
        self.mapper = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )
        # self.mapper = nn.Linear(self.embed_dim, num_classes)
        self.output_size = self.chw[1:]
        # self.focal_loss = FocalLoss()
        # self.ce_loss = torch.nn.CrossEntropyLoss()
        self.cedice_loss = CEDiceLoss()
    def forward(self, x):
        with torch.no_grad():
            embeddings = self.backbone.forward_features(x)
            x = embeddings["x_norm_patchtokens"]
        x = self.mapper(x)
        x = rearrange(x, "b (ph pw) k -> b k ph pw", ph=16)
        # bilinear upsample
        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return x

    def training_step(self, batch, batch_idx):
        images, masks = batch["original_images"], batch["original_segmentation_maps"]
        masks = rearrange(masks, "b 1 h w -> b h w").long()
        # print(images.shape, masks.shape) # torch.Size([2, 3, 224, 224]) torch.Size([2, 1, 224, 224])
        x = self.forward(images)
        # x shape is (b num_classes h w), masks shape is (b cls h w)
        # calculate cross entropy loss
        # ce_loss = torch.nn.CrossEntropyLoss()
        loss = self.cedice_loss(x, masks)
        self.log("trainLoss", loss, sync_dist=True, batch_size=BATCH_SIZE)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch["original_images"], batch["original_segmentation_maps"]
        masks = rearrange(masks, "b 1 h w -> b h w").long()
        # print(images.shape, masks.shape) # torch.Size([2, 3, 224, 224]) torch.Size([2, 1, 224, 224])
        x = self.forward(images)
        # x shape is (b num_classes h w), masks shape is (b cls h w)
        # calculate cross entropy loss
        loss = self.cedice_loss(x, masks)
        self.log("valLoss", loss, sync_dist=True, batch_size=BATCH_SIZE, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # AdamW optimizer with specified learning rate
        # optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr, weight_decay=0.0005)
        optimizer = torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=0.0005)
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return {'optimizer': optimizer}
    
if __name__ == "__main__":
    device = "cuda"
    dm = SegmentationDataModule(dataset_dir="/mnt/data/Finetune_mask2Former/data/dataset", img_size=(224, 224), batch_size=2, num_workers=2)
    # x = torch.randn((2, 3, 224, 224)).to(device)
    model = DinoSegmentation(chw=[3, 224, 224], num_classes=2).to(device)

    trainer = Trainer(
        logger=LOGGER,
        accelerator='cuda',
        devices=[0],
        strategy="auto",
        callbacks=[CHECKPOINT_CALLBACK],
        max_epochs=100
    )
    trainer.fit(model, datamodule=dm)
    # trainer.validate(model, datamodule=dm)
    # pred = model(
    # print(pred.shape)
    print("saving model!")
    trainer.save_checkpoint("/home/patrick/workspace/cv/pat-cv-lifecycle/checkpoints/dino_linear_decoder.ckpt")