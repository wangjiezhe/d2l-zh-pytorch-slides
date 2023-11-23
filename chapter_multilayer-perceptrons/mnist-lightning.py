#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from lightning import LightningModule, Trainer
from torchmetrics.functional import accuracy

torch.set_float32_matmul_precision('high')


class MNISTModel(LightningModule):
  def __init__(self, batch_size=256, num_hiddens=1024, lr=0.5, data_dir="../data", num_workers=8):
    super().__init__()
    self.save_hyperparameters()
    self.net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, self.hparams.num_hiddens),
                             nn.ReLU(), nn.Linear(self.hparams.num_hiddens, 10))
    self.loss = nn.CrossEntropyLoss()
    self.trans = ToTensor()

  def forward(self, X):
    return self.net(X)

  def training_step(self, batch):
    X, y = batch
    y_hat = self.net(X)
    train_loss = self.loss(y_hat, y)
    self.log("train_loss", train_loss, prog_bar=True)
    return train_loss

  def test_step(self, batch):
    X, y = batch
    y_hat = self.net(X)
    test_loss = self.loss(y_hat, y)
    test_acc = accuracy(y_hat, y, task="multiclass", num_classes=10, average="micro")
    self.log("test_loss", test_loss, prog_bar=True)
    self.log("test_acc", test_acc, prog_bar=True)
    return test_loss

  def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)

  def prepare_data(self):
    MNIST(root=self.hparams.data_dir, download=True)

  def setup(self, stage=None):
    self.mnist_train = MNIST(root=self.hparams.data_dir, train=True, transform=self.trans)
    self.mnist_test = MNIST(root=self.hparams.data_dir, train=False, transform=self.trans)

  def train_dataloader(self):
    return DataLoader(self.mnist_train, self.hparams.batch_size,
                      num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)

  def test_dataloader(self):
    return DataLoader(self.mnist_test, self.hparams.batch_size,
                      num_workers=self.hparams.num_workers, pin_memory=True, shuffle=False)

if __name__ == "__main__":
  model = MNISTModel()
  trainer = Trainer(max_epochs=10, log_every_n_steps=1)
  trainer.fit(model)
  trainer.test(model)