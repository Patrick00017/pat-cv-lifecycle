import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader 
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from einops import rearrange, pack
import matplotlib.pyplot as plt

class ImageSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if not f.startswith('.')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.filenames[idx] + '.jpg')
        mask_path = os.path.join(self.masks_dir, self.filenames[idx] + '.png') 
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        mask = Image.open(mask_path)
        # mask=np.array(mask)
        # mask[mask==255]=1
        # mask = mask[np.newaxis, :, :]
        # print(mask.shape)
        mask = self.mask_transform(mask)
        # np_image is [0-255], np_mask is [0|1]
        return image, mask
    
class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, img_size, batch_size, num_workers):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.img_size = img_size
        self.transform = T.Compose([
            T.Lambda(self._resize_to_target),  # 调整尺寸
            T.ToTensor(),                      # 转换为张量
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    # 标准化
        ])
        self.mask_transform = T.Compose([
            T.Lambda(self._resize_mask),  # 调整尺寸
            T.ToTensor(),                      # 转换为张量
        ])

    def _resize_to_target(self, img):
        """调整图像到目标尺寸，保持宽高比"""
        return F.resize(img, self.img_size)
    def _resize_mask(self, mask):
        return F.resize(mask, self.img_size, interpolation=InterpolationMode.NEAREST)
    def setup(self, stage=None):
        # print(self.img_size)
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageSegmentationDataset(images_dir=os.path.join(self.dataset_dir, 'images', 'train'),
                                                          masks_dir=os.path.join(self.dataset_dir, 'labels', 'train'),
                                                          transform=self.transform, mask_transform=self.mask_transform) # Add your transforms here
            self.val_dataset = ImageSegmentationDataset(images_dir=os.path.join(self.dataset_dir, 'images', 'val'),
                                                        masks_dir=os.path.join(self.dataset_dir, 'labels', 'val'),
                                                        transform=self.transform, mask_transform=self.mask_transform) # Add your transforms here
        if stage == 'test' or stage is None:
            self.test_dataset = ImageSegmentationDataset(images_dir=os.path.join(self.dataset_dir, 'images', 'test'),
                                                         masks_dir=os.path.join(self.dataset_dir, 'labels', 'test'),
                                                         transform=self.transform, mask_transform=self.mask_transform) # Add your transforms here
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
        
    def collate_fn(self,batch):
        inputs = list(zip(*batch))
        images=inputs[0]
        # print(images[0].shape)
        segmentation_maps=inputs[1]
        batch = {}
        batch["original_images"] = pack(images, "* c h w")[0]
        batch["original_segmentation_maps"] = pack(segmentation_maps, "* c h w")[0]
        return batch
    
if __name__ == "__main__":
    pass
    # data_module = SegmentationDataModule(
    #     dataset_dir="/mnt/data/Finetune_mask2Former/data/dataset", 
    #     img_size=(640, 640),
    #     batch_size=2, 
    #     num_workers=1
    # )
    # data_module.setup(stage="fit")
    # train_loader = data_module.train_dataloader()
    # batch = next(iter(train_loader))
    # print(batch["original_images"].shape)
    # print(batch["original_segmentation_maps"].shape)
    # # print(batch["original_images"])
    # # print(batch["original_segmentation_maps"])
    
    # images, masks = batch["original_images"], batch["original_segmentation_maps"]
    # for i in range(len(images)):
    #     img = images[i]
    #     img = rearrange(img, "c h w -> h w c")
    #     mask = masks[i]
    #     mask = rearrange(mask, "c h w -> h w c")

    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(img)
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(mask)
    #     plt.show()