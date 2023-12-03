#!/usr/bin/env python3

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

torch.set_float32_matmul_precision("high")


class MNISTModel(LightningModule):
    def __init__(
        self, batch_size=256, num_hiddens=1024, lr=0.5, data_dir="../data", num_workers=8
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_hiddens = num_hiddens
        self.lr = lr
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, self.num_hiddens * 2),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(self.num_hiddens * 2, 10),
        )
        self.net.apply(self.init_weight)
        self.loss = nn.CrossEntropyLoss()
        self.trans = ToTensor()

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, X):
        return self.net(X)

    def training_step(self, batch):
        X, y = batch
        y_hat = self.net(X)
        train_loss = self.loss(y_hat, y)
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch):
        X, y = batch
        y_hat = self.net(X)
        val_loss = self.loss(y_hat, y)
        val_acc = accuracy(y_hat, y, task="multiclass", num_classes=10, average="micro")
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

    def test_step(self, batch):
        X, y = batch
        y_hat = self.net(X)
        test_loss = self.loss(y_hat, y)
        test_acc = accuracy(y_hat, y, task="multiclass", num_classes=10, average="micro")
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=1e-4)

    def prepare_data(self):
        MNIST(root=self.data_dir, download=True)

    def setup(self, stage=None):
        self.train_data, self.val_data = random_split(
            MNIST(root=self.data_dir, train=True, transform=self.trans), [0.8, 0.2]
        )
        self.test_data = MNIST(root=self.data_dir, train=False, transform=self.trans)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )


if __name__ == "__main__":
    model = MNISTModel()
    trainer = Trainer(max_epochs=100, log_every_n_steps=1)
    trainer.fit(model)
    trainer.test(model)
