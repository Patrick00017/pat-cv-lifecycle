# Pat cv lifecycle

This is a project from Patrick Huang, to maintain my important experience in computer vision and deep learning area.

Have Fun. Love **yourself** first.


## Project Structure
```
your_cv_project/
│
├── configs/               # 配置文件目录 (YAML 或 Python)
│   └── train_config.yaml
│
├── data/                  # 数据相关（或符号链接到数据集）
│   ├── datasets/
│   └── transforms.py      # 自定义数据增强
│
├── src/                   # 主要源代码
│   ├── __init__.py
│   ├── data/              # 数据模块
│   │   ├── __init__.py
│   │   ├── datamodules.py # LightningDataModule
│   │   └── datasets.py    # 自定义 Dataset
│   │
│   ├── models/            # 模型模块
│   │   ├── __init__.py
│   │   ├── lit_models.py  # LightningModule
│   │   └── torch_models.py # 纯 PyTorch nn.Module
│   │
│   ├── engine/            # 训练/验证逻辑（通常已在 LM 中，复杂时可提取）
│   │   └── callbacks.py   # 自定义 Callbacks
│   │
│   └── utils/             # 工具函数
│       ├── __init__.py
│       ├── logging.py
│       └── helpers.py
│
├── scripts/               # 用于执行、部署的脚本
│   ├── train.py
│   ├── test.py
│   ├── export.py          # 导出为 TorchScript/ONNX
│   └── deploy_infer.py    # 部署后的推理示例
│
├── logs/                  # 日志和实验记录（由 Lightning 自动生成）
│   ├── tensorboard/       # TensorBoard 日志
│   └── checkpoints/       # 模型检查点
│
├── requirements.txt       # Python 依赖
└── README.md              # 项目说明、复现步骤
```

set include path to [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for instant ngp. Thanks for amazing work.

## Modules
1. VIT transformer
2. Masked Auto Encoder
3. NeRF
4. Dino Linear Probe Segmentation

## Plan
1. InstantNGP
2. TensoRF
3. PixelNeRF