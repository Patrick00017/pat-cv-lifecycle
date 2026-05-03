from ultralytics import YOLO

# 1. 加载您的模型
# 请将 'yolo.pt' 替换为您实际训练好的模型路径
model = YOLO("best.pt")

# 2. 导出为 ONNX 格式
# opset=12 是 X-AnyLabeling 兼容性的参数
# 如果您使用 CPU 环境，请删除 device=0 参数
success = model.export(format="onnx", opset=12)

print(f"导出状态: {success}")
# 导出成功后，会在同目录下生成对应的 .onnx 文件
