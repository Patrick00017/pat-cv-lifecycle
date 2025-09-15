from models import DinoSegmentation
from einops import rearrange
from data.image_segmentation_dataset import SegmentationDataModule
import matplotlib.pyplot as plt
import torch

device = "cuda"
dm = SegmentationDataModule(dataset_dir="/mnt/data/Finetune_mask2Former/data/dataset", img_size=(224, 224), batch_size=2, num_workers=2)
# x = torch.randn((2, 3, 224, 224)).to(device)
# model = DinoSegmentation(chw=[3, 224, 224], num_classes=2).to(device)
model = DinoSegmentation.load_from_checkpoint("dino_linear_decoder.ckpt", chw=[3, 224, 224], num_classes=2).to(device)
model.eval()

dm.setup(stage="test")
loader = dm.test_dataloader()
for batch in loader:
    images, masks = batch["original_images"], batch["original_segmentation_maps"]
    masks = masks.int()
    print(torch.max(masks), torch.min(masks))
    input_tensor = images.to(device)
    preds = model.forward(input_tensor)
    preds = torch.softmax(preds, dim=1)
    pred_masks = preds.argmax(dim=1).cpu()
    pred_masks = pred_masks.int()
    print(torch.max(pred_masks), torch.min(pred_masks))
    pred_masks = rearrange(pred_masks, "b h w -> b 1 h w")
    # print(pred_masks.shape)
    for i in range(len(images)):
        img = images[i]
        img = rearrange(img, "c h w -> h w c")
        mask = masks[i]
        mask = rearrange(mask, "c h w -> h w c")
        pred_mask = pred_masks[i]
        pred_mask = rearrange(pred_mask, "c h w -> h w c")
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask)
        plt.show()