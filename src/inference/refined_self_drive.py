from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import torch
from torchvision import transforms
from models.self_driving_cnn_model import SelfDrivingCNNModel
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"
socketio = SocketIO(app, cors_allowed_origins="*")
speed_limit = 10

model = SelfDrivingCNNModel()
weights = torch.load("D:/code/pat-cv-lifecycle/checkpoints/drive.ckpt")
model.load_state_dict(weights["state_dict"])
model.eval()
print("load model done.")


def img_process(img):
    # img need to be PIL Image type
    transform = transforms.Compose(
        [
            transforms.CenterCrop(size=(66, 200)),
            transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
        ]
    )
    return transform(img)


@socketio.on("disconnect")
def disconnect_msg():
    print("client disconnected.")


@socketio.on("telemetry")
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


@socketio.on("connect")
def connect(sid, environ):
    print("Connected")
    send_control(0, 0)


def send_control(steering_angle, throttle):
    socketio.emit(
        "steer",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__(),
        },
    )


if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=4567)
