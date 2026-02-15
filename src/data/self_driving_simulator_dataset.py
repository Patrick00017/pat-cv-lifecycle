from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import os
from torch.utils.data import Dataset
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import ntpath
from PIL import Image
from torch.utils.data import random_split


def pathleaf(path):
    head, tail = ntpath.split(path)
    return tail


class SelfDrivingDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.dir = data_dir
        self.image_dir_path = os.path.join(data_dir, "IMG")
        self.transform = transform
        columns = [
            "center",
            "left",
            "right",
            "steering",
            "throttle",
            "reverse",
            "speed",
        ]
        self.data = pd.read_csv(
            os.path.join(self.dir, "driving_log.csv"), names=columns
        )

        # prepare the data distribution
        self.prepare_samples()

    def load_img_steering(self):
        datadir = self.dir
        imagedir = self.image_dir_path
        df = self.data
        image_path = []
        steering = []
        for i in range(len(df)):
            indexed_data = df.iloc[i]
            center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
            image_path.append(os.path.join(imagedir, center.strip()))
            steering.append(float(indexed_data[3]))
            image_path.append(os.path.join(imagedir, left.strip()))
            steering.append(
                float(indexed_data[3]) + 0.15
            )  # left image need to automatic turn right
            image_path.append(os.path.join(imagedir, right.strip()))
            steering.append(
                float(indexed_data[3]) - 0.15
            )  # and right image need to turn left
        # image_paths = np.asarray(image_path)
        # steerings = np.asarray(steering)
        return image_path, steering

    def prepare_samples(self):
        """
        prepare all the samples for training,
        data is stored in self.image_path_list, self.steering_list
        """
        # trim filename
        self.data["center"] = self.data["center"].apply(pathleaf)
        self.data["left"] = self.data["left"].apply(pathleaf)
        self.data["right"] = self.data["right"].apply(pathleaf)
        # prepare the samples
        num_bins = 25
        samples_per_bin = 400
        hist, bins = np.histogram(self.data["steering"], num_bins)
        remove_list = []
        for j in range(num_bins):
            list_ = []
            for i in range(len(self.data["steering"])):
                if (
                    self.data["steering"][i] >= bins[j]
                    and self.data["steering"][i] <= bins[j + 1]
                ):
                    list_.append(i)
            list_ = shuffle(list_)
            list_ = list_[samples_per_bin:]
            remove_list.extend(list_)
        print("Removed:", len(remove_list))
        self.data.drop(self.data.index[remove_list], inplace=True)
        print("Remaining:", len(self.data))
        self.image_path_list, self.steering_list = self.load_img_steering()

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        img_path = self.image_path_list[idx]
        steering = self.steering_list[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, steering


class SelfDrivingDataModule(pl.LightningModule):
    def __init__(
        self, batch_size, dataset_dir="D:\\Games\\beta_simulator_windows\\dataset"
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.train_transform = transforms.Compose(
            [
                # transforms.Resize(size=(66, 200)),  # 随机裁剪并缩放至224x224
                transforms.CenterCrop(size=(66, 200)),
                # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                # transforms.RandomRotation(15),  # 随机旋转±15度
                # transforms.ColorJitter(  # 随机调整颜色
                #     brightness=0.2, contrast=0.2, saturation=0.2
                # ),
                transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
                # transforms.Normalize(  # 标准化
                #     mean=[0.485, 0.456, 0.406],  # ImageNet的均值和标准差
                #     std=[0.229, 0.224, 0.225],
                # ),
            ]
        )

    def setup(self, stage=None):
        full_dataset = SelfDrivingDataset(self.dataset_dir, self.train_transform)
        # 计算分割大小
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)  # 80% 训练
        test_size = total_size - train_size  # 20% 测试
        # 随机分割
        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),  # 设置随机种子
        )
        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = test_dataset
        if stage == "test" or stage is None:
            self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
