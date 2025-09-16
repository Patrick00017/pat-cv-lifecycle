import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from dinov2.models.vision_transformer import vit_small
# from dinov2.data.transforms import make_classification_eval_transform
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from einops import rearrange
from data.image_segmentation_dataset import SegmentationDataModule
from pytorch_lightning import Trainer
from models.losses import CEDiceLoss

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

def resize(input_data,
       size=None,
       scale_factor=None,
       mode='nearest',
       align_corners=None,
       warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    print(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input_data, size, scale_factor, mode, align_corners)

class BNHead(nn.Module):
    """Just a batchnorm."""
    def __init__(self, num_classes=2, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        # HARDCODED IN_CHANNELS FOR NOW.
        self.in_channels = 1536
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors
        self.in_index = [0, 1, 2, 3]
        self.input_transform = 'resize_concat'
        self.align_corners = False
        self.conv_seg = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.bn(x)
        return feats
    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input_data=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs
    def cls_seg(self, feat):
        """Classify each pixel."""
        output = self.conv_seg(feat)
        return output
        
    
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

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
        self.head = BNHead(num_classes=num_classes)

        self.output_size = self.chw[1:]
        self.cedice_loss = CEDiceLoss()
    def forward(self, x):
        with torch.no_grad():
            x = self.backbone.get_intermediate_layers(x, n=4)
        x = self.mapper(x)
        x = [rearrange(t, "b (ph pw) e -> b e ph pw", ph=16) for t in x]
        x = self.mapper(x)
        x = nn.functional.interpolate(
            x, size=self.output_size, 
            mode="bilinear", 
            align_corners=False
        )
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