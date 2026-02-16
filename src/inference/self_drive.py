import socketio
import eventlet
import numpy as np
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2
import torch
from torchvision import transforms
from models.self_driving_cnn_model import SelfDrivingCNNModel


def img_process(img):
    # img need to be PIL Image type
    transform = transforms.Compose(
        [
            transforms.CenterCrop(size=(66, 200)),
            transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
        ]
    )
    return transform(img)


sio = socketio.Server(cors_allowed_origins="*")

app = Flask(__name__)  #'__main__'
# app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)  # 正确包装WSGI应用
speed_limit = 10

model = SelfDrivingCNNModel()
weights = torch.load("D:/code/pat-cv-lifecycle/checkpoints/drive.ckpt")
model.load_state_dict(weights["state_dict"])
model.eval()
print("load model done.")


@sio.on("telemetry")
def telemetry(sid, data):
    print("telemetry")
    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = img_process(image)
    print(f"Get data done. speed: {speed}")
    # image = np.array([image])
    steering_angle = float(model(image))
    throttle = 1.0 - speed / speed_limit
    print("{} {} {}".format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
    print("send control...")


@sio.on("connect")
def connect(sid, environ):
    print("Connected")
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__(),
        },
    )


if __name__ == "__main__":
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
