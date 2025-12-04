from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision
import os
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

# pycocotools is a required dependency
from pycocotools.coco import COCO
import os
import json
from PIL import Image
from collections import defaultdict


data_root_path = "datasets/"

# def get_mnist_train_loader(batch_size: int):
#     transform = transforms.ToTensor()
#     dataset = MNIST(data_root_path, download=True, transform=transform)
#     train_loader = DataLoader(dataset, batch_size=batch_size)
#     return train_loader


class MNISTDataModule(pl.LightningModule):
    def __init__(self, batch_size, dataset_dir=data_root_path):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def setup(self, stage=None):
        # print(self.img_size)
        if stage == "fit" or stage is None:
            self.train_dataset = MNIST(
                self.dataset_dir, download=True, transform=self.transform, train=True
            )
            self.val_dataset = MNIST(
                self.dataset_dir, download=True, transform=self.transform, train=False
            )
        if stage == "test" or stage is None:
            self.test_dataset = MNIST(
                self.dataset_dir, download=True, transform=self.transform, train=False
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class Cifar10DataModule(pl.LightningModule):
    def __init__(self, batch_size, dataset_dir=data_root_path):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def setup(self, stage=None):
        # print(self.img_size)
        if stage == "fit" or stage is None:
            self.train_dataset = CIFAR10(
                root=self.dataset_dir,
                train=True,
                download=True,
                transform=self.train_transform,
            )
            self.val_dataset = CIFAR10(
                root=self.dataset_dir,
                train=False,
                download=True,
                transform=self.test_transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10(
                root=self.dataset_dir,
                train=False,
                download=True,
                transform=self.test_transform,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class ContinuousCocoDataset(VisionDataset):
    """
    A PyTorch Dataset for COCO-style data that supports continuous learning.

    This class can load and merge multiple COCO annotation files. This is useful
    for incremental learning scenarios where you start with a base dataset and
    add new data over time without changing the existing annotation files.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_files (list[string]): List of paths to COCO annotation files.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, ann_files, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_files = ann_files
        self.coco = self._load_and_merge_annotations(ann_files)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_and_merge_annotations(self, ann_files):
        """
        Loads and merges multiple COCO annotation files, re-indexing images
        and annotations to prevent ID collisions.
        """
        if not ann_files:
            raise ValueError("Annotation files list cannot be empty.")

        # Load the first file to initialize the dataset structure
        with open(ann_files[0], "r") as f:
            base_json = json.load(f)

        merged_dataset = {
            "info": base_json.get("info", {}),
            "licenses": base_json.get("licenses", []),
            "categories": base_json.get("categories", []),
            "images": [],
            "annotations": [],
        }

        # Use counters for new unique IDs
        img_id_counter = 1
        ann_id_counter = 1

        for ann_file in ann_files:
            with open(ann_file, "r") as f:
                data = json.load(f)

            # --- Remap image and annotation IDs to be unique ---
            # Create a mapping from old image IDs in the current file to new unique IDs
            old_to_new_img_id = {
                img["id"]: img_id_counter + i for i, img in enumerate(data["images"])
            }

            # Add images with new IDs
            for img in data["images"]:
                new_img = img.copy()
                new_img["id"] = old_to_new_img_id[img["id"]]
                merged_dataset["images"].append(new_img)

            # Add annotations with new IDs
            if "annotations" in data:
                for ann in data["annotations"]:
                    new_ann = ann.copy()
                    new_ann["id"] = ann_id_counter
                    new_ann["image_id"] = old_to_new_img_id[ann["image_id"]]
                    merged_dataset["annotations"].append(new_ann)
                    ann_id_counter += 1

            img_id_counter += len(data["images"])

        coco = COCO()
        coco.dataset = merged_dataset
        coco.createIndex()
        return coco

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Get image info
        img_info = coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        img_path = os.path.join(self.root, path)

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Create target
        target = {
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "boxes": [],
            "labels": [],
            "area": [],
            "iscrowd": [],
        }

        # Bounding boxes are [x, y, width, height]
        boxes = [obj["bbox"] for obj in anns]
        # Convert to [x_min, y_min, x_max, y_max]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]

        target["boxes"] = boxes
        target["labels"] = torch.tensor(
            [obj["category_id"] for obj in anns], dtype=torch.int64
        )
        target["area"] = torch.tensor(
            [obj["area"] for obj in anns], dtype=torch.float32
        )
        target["iscrowd"] = torch.tensor(
            [obj.get("iscrowd", 0) for obj in anns], dtype=torch.int64
        )

        if self.transform is not None:
            img, target = self.transform(img, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy data and annotation files for demonstration

    # Create dummy directories
    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("dataset/annotations", exist_ok=True)

    # Create dummy images
    Image.new("RGB", (100, 100), color="red").save("dataset/images/img1.png")
    Image.new("RGB", (100, 100), color="blue").save("dataset/images/img2.png")
    Image.new("RGB", (100, 100), color="green").save("dataset/images/img3.png")

    # --- Create first annotation file (represents initial dataset) ---
    coco_data_1 = {
        "images": [
            {"id": 1, "file_name": "images/img1.png", "height": 100, "width": 100},
            {"id": 2, "file_name": "images/img2.png", "height": 100, "width": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 50, 50],
                "area": 2500,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 2,
                "bbox": [20, 20, 30, 30],
                "area": 900,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
        ],
    }
    with open("dataset/annotations/data_1.json", "w") as f:
        json.dump(coco_data_1, f)

    # --- Create second annotation file (represents new data) ---
    # Note: Image and Annotation IDs start from 1 again, which the loader must handle.
    coco_data_2 = {
        "images": [
            {"id": 1, "file_name": "images/img3.png", "height": 100, "width": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [5, 5, 25, 25],
                "area": 625,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
        ],
    }
    with open("dataset/annotations/data_2.json", "w") as f:
        json.dump(coco_data_2, f)

    print("Created dummy data and annotation files in 'dataset/' directory.")

    # 1. Load only the initial dataset
    print("\n--- Loading initial dataset ---")
    initial_ann_file = ["dataset/annotations/data_1.json"]
    initial_dataset = ContinuousCocoDataset(root="dataset", ann_files=initial_ann_file)
    print(f"Initial dataset size: {len(initial_dataset)}")

    # 2. Load the combined dataset for "continuous learning"
    print("\n--- Loading combined dataset for continuous learning ---")
    combined_ann_files = [
        "dataset/annotations/data_1.json",
        "dataset/annotations/data_2.json",
    ]
    combined_dataset = ContinuousCocoDataset(
        root="dataset", ann_files=combined_ann_files
    )
    print(f"Combined dataset size: {len(combined_dataset)}")

    # Verify an item from the combined dataset
    img, target = combined_dataset[2]  # This should be img3.png
    print("\n--- Verifying a sample from the combined dataset ---")
    print(f"Image ID from target: {target['image_id'].item()}")
    print(f"Target boxes: {target['boxes']}")
    print(f"Target labels: {target['labels']}")

    # Check if the image IDs are correctly re-indexed
    # The third image (img3.png) had an original ID of 1 in its own file,
    # but should now have a new, unique ID in the merged dataset.
    img_info = combined_dataset.coco.loadImgs(target["image_id"].item())[0]
    print(f"Image file name from COCO API: {img_info['file_name']}")
    assert img_info["file_name"] == "images/img3.png"
    print(
        "Assertion successful: Image ID correctly re-indexed and maps to the right file."
    )

    # Example with a DataLoader
    print("\n--- Using with DataLoader ---")
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader = DataLoader(
        combined_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )

    images, targets = next(iter(data_loader))
    print(f"Batch contains {len(images)} images.")
    print(f"First target in batch: {targets[0]}")
