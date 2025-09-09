import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from models.simple_autoencoder import SimpleAutoEncoder
from data.common_datasets import get_mnist_train_loader
from pprint import pprint
from tqdm import tqdm
import yaml
import os

# read yaml config file
with open('configs/simple_autoencoder.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

dataloader = get_mnist_train_loader(cfg["train"]["batch_size"])

model = SimpleAutoEncoder(cfg)

# train
# trainer = L.Trainer()
# trainer.fit(model=autoencoder, train_dataloaders=dataloader)

optimizer = model.configure_optimizers()
for i in tqdm(range(cfg["train"]["epoch"])):
    total_loss = 0.0
    num_batch = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        loss = model.training_step(batch, batch_idx)
        loss.backward()
        total_loss += loss.detach().item()
        optimizer.step()
        optimizer.zero_grad()
    
    pprint(f"epoch: {i}, train_loss: {total_loss / num_batch}")

# save last ckpt
output_dir = cfg["train"]["output_dir"]
model_name = cfg["model"]["name"]
final_dir_path = f"{output_dir}/{model_name}"
if not os.path.exists(final_dir_path):
    os.makedirs(final_dir_path)
torch.save(model.state_dict(), f"{final_dir_path}/last_checkpoint.pth")
pprint(f"train process complete. save checkpoint at {final_dir_path}/last_checkpoint.pth")