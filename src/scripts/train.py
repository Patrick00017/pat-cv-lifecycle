import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from models.simple_auto_encoder import SimpleAutoEncoder
from data.common_datasets import get_mnist_train_loader

dataloader = get_mnist_train_loader()

autoencoder = SimpleAutoEncoder()

# train
# trainer = L.Trainer()
# trainer.fit(model=autoencoder, train_dataloaders=dataloader)

optimizer = autoencoder.configure_optimizers()
for batch_idx, batch in enumerate(dataloader):
    loss = autoencoder.training_step(batch, batch_idx)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()