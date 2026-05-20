from ultralytics import YOLO

# Load a model
model = YOLO(
    "D:/code/pat-cv-lifecycle/src/examples/yolo/best (1).pt"
)  # load a pretrained model (recommended for training)
# Train the model
results = model.train(data="D:/datasets/warp_all_line/warp.yaml", epochs=100, imgsz=640)
