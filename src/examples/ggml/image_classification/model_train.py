import gguf
import numpy as np
from torch.autograd import Variable
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import sys
from time import time

data_dir = "data/"
device = "cpu"
input_size = (224, 224)
# hparams
num_classes = 2  # number of output classes discrete range [0,9]
num_epochs = (
    30  # number of times which the entire dataset is passed throughout the model
)
batch_size = 64  # the size of input data used for one iteration
lr = 1e-3  # size of step

# train_transforms = transforms.Compose(
#     [
#         transforms.RandomRotation(30),
#         transforms.RandomResizedCrop(input_size[0]),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

# test_transforms = transforms.Compose(
#     [
#         transforms.Resize(255),
#         transforms.CenterCrop(input_size[0]),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

# train_data = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
# test_data = datasets.ImageFolder(data_dir + "/test", transform=test_transforms)

# trainloader = torch.utils.data.DataLoader(
#     train_data, batch_size=batch_size, shuffle=True
# )
# testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# load pretrained densenet121
# model = models.densenet121(weights=None)
# # models.DenseNet121_Weights()
# pre = torch.load("weights/densenet121-a639ec97.pth")
# model.load_state_dict(pre)
# model = torch.hub.load("pytorch/vision", "densenet121", weights="IMAGENET1K_V2")
# model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
model = models.resnet50(pretrained=True)
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# define the classifer
from collections import OrderedDict

classifier = nn.Sequential(
    OrderedDict(
        [
            ("fc1", nn.Linear(1000, 512)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(512, 256)),
            ("relu2", nn.ReLU()),
            ("fc3", nn.Linear(256, num_classes)),
            ("output", nn.LogSoftmax(dim=1)),
        ]
    )
)
model.classifier = classifier
print(model)
# optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
# criterion = nn.NLLLoss()

# traininglosses = []
# testinglosses = []
# testaccuracy = []
# totalsteps = []
# steps = 0
# running_loss = 0
# print_every = 5
# for epoch in range(num_epochs):
#     for inputs, labels in trainloader:
#         steps += 1
#         # Move input and label tensors to the default device
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()

#         logps = model.forward(inputs)
#         loss = criterion(logps, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#         if steps % print_every == 0:
#             test_loss = 0
#             accuracy = 0
#             model.eval()
#             with torch.no_grad():
#                 for inputs, labels in testloader:
#                     inputs, labels = inputs.to(device), labels.to(device)
#                     logps = model.forward(inputs)
#                     batch_loss = criterion(logps, labels)

#                     test_loss += batch_loss.item()

#                     # Calculate accuracy
#                     ps = torch.exp(logps)
#                     top_p, top_class = ps.topk(1, dim=1)
#                     equals = top_class == labels.view(*top_class.shape)
#                     accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

#             traininglosses.append(running_loss / print_every)
#             testinglosses.append(test_loss / len(testloader))
#             testaccuracy.append(accuracy / len(testloader))
#             totalsteps.append(steps)
#             print(
#                 f"Device {device}.."
#                 f"Epoch {epoch+1}/{num_epochs}.. "
#                 f"Step {steps}.. "
#                 f"Train loss: {running_loss/print_every:.3f}.. "
#                 f"Test loss: {test_loss/len(testloader):.3f}.. "
#                 f"Test accuracy: {accuracy/len(testloader):.3f}"
#             )
#             running_loss = 0
#             model.train()

model_path = "weights/icls.gguf"
gguf_writer = gguf.GGUFWriter(model_path, "icls")
print()
print(f"Model tensors saved to {model_path}:")
for tensor_name in model.state_dict().keys():
    data = model.state_dict()[tensor_name].squeeze().cpu().numpy()
    print(tensor_name, "\t", data.shape)
    gguf_writer.add_tensor(tensor_name, data)
gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()
gguf_writer.close()
