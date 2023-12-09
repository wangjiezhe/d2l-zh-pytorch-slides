#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from scipy import optimize
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

torch.set_float32_matmul_precision("high")


class MNISTDataModel(pl.LightningDataModule):
    def __init__(self, batch_size=256, data_dir="../data", num_workers=8):
        super().__init__()
        # self.save_hyperparameters()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.trans = ToTensor()

    def prepare_data(self):
        MNIST(root=self.data_dir, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data, self.val_data = random_split(
                MNIST(root=self.data_dir, train=True, transform=self.trans), [0.8, 0.2]
            )
        elif stage == "test" or stage is None:
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


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=10, average="micro")
        self.val_acc = Accuracy(task="multiclass", num_classes=10, average="micro")
        self.test_acc = Accuracy(task="multiclass", num_classes=10, average="micro")

    def forward(self, X):
        return self.net(X)

    def training_step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = self.loss(y_hat, y)
        acc = self.train_acc(y_hat, y)
        metrics = {"train_loss": loss, "train_acc": acc}
        self.log_dict(metrics, prog_bar=True)
        return loss

    def validation_step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = self.loss(y_hat, y)
        acc = self.val_acc(y_hat, y)
        metrics = {"val_loss": loss, "val_acc": acc}
        self.log_dict(metrics, prog_bar=True)
        # self.log("hp_metric", acc)
        return loss

    def test_step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = self.loss(y_hat, y)
        acc = self.test_acc(y_hat, y)
        metrics = {"test_loss": loss, "test_acc": acc}
        self.log_dict(metrics, prog_bar=True)
        # self.log("hp_metric", acc)
        return loss

    def on_test_end(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/train_acc": self.train_acc.compute(),
                "hp/val_acc": self.val_acc.compute(),
                "hp/test_acc": self.test_acc.compute(),
            },
        )

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)
        return optimizer(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )


class LeNet5Model(MNISTModel):
    def __init__(
        self,
        lr=0.05,
        weight_decay=1e-4,
        optimizer="SGD",
        activation="Tanh",
        pool="avg",
        c1_channel=6,
        c1_kernel=5,
        c1_padding=2,
        s2_kernel=2,
        c3_channel=16,
        c3_kernel=5,
        c3_padding=0,
        s4_kernel=2,
        f5_size=120,
        f6_size=84,
        output_size=10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.f5_input_size = (
            c3_channel
            * (((28 - (c1_kernel - c1_padding * 2) + 1) // s2_kernel - c3_kernel + 1) // s4_kernel)
            ** 2
        )
        activation = getattr(nn, self.hparams.activation)
        if self.hparams.pool == "avg":
            pool = nn.AvgPool2d
        elif self.hparams.pool == "max":
            pool = nn.MaxPool2d

        self.net = nn.Sequential(
            nn.Conv2d(
                1,
                self.hparams.c1_channel,
                kernel_size=self.hparams.c1_kernel,
                padding=self.hparams.c1_padding,
            ),
            activation(),
            pool(kernel_size=self.hparams.s2_kernel),
            nn.Conv2d(
                self.hparams.c1_channel,
                self.hparams.c3_channel,
                kernel_size=self.hparams.c3_kernel,
            ),
            activation(),
            pool(kernel_size=self.hparams.s4_kernel),
            nn.Flatten(),
            nn.Linear(self.f5_input_size, self.hparams.f5_size),
            nn.Linear(self.hparams.f5_size, self.hparams.f6_size),
            activation(),
            nn.Linear(self.hparams.f6_size, self.hparams.output_size),
        )


def lenet_5_traditional():
    pl.seed_everything(42)

    ck_train_loss = ModelCheckpoint(
        monitor="train_loss", filename="model-{epoch:02d}-{train_loss:.2f}"
    )
    ck_train_acc = ModelCheckpoint(
        monitor="train_acc", mode="max", filename="model-{epoch:02d}-{val_acc:.2f}"
    )
    ck_val_loss = ModelCheckpoint(monitor="val_loss", filename="model-{epoch:02d}-{val_loss:.2f}")
    ck_val_acc = ModelCheckpoint(
        monitor="val_acc", mode="max", filename="model-{epoch:02d}-{val_acc:.2f}"
    )

    data = MNISTDataModel()
    model = LeNet5Model(
        activation="Tanh",
        pool="avg",
        weight_decay=0,
    )
    logger = TensorBoardLogger(".", default_hp_metric=False, version="lenet_5_traditional")
    trainer = pl.Trainer(
        max_epochs=1000,
        logger=logger,
        callbacks=[ck_train_loss, ck_train_acc, ck_val_loss, ck_val_acc],
    )
    tuner = Tuner(trainer)

    tuner.lr_find(model, data, min_lr=1e-3, max_lr=1e-1, num_training=1000)
    trainer.fit(model, data)
    trainer.test(model, data)

    trainer = pl.Trainer(logger=None)
    model_best = LeNet5Model.load_from_checkpoint(ck_val_acc.best_model_path)
    trainer.test(model_best, data)


class LeNet5ModernModel(LeNet5Model):
    def __init__(self):
        super().__init__(activation="ReLU", pool="max", weight_decay=1e-3)
        self.net.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")


def lenet_5_modern():
    pl.seed_everything(42)

    ck_train_loss = ModelCheckpoint(
        monitor="train_loss", filename="model-{epoch:02d}-{train_loss:.2f}"
    )
    ck_train_acc = ModelCheckpoint(
        monitor="train_acc", mode="max", filename="model-{epoch:02d}-{val_acc:.2f}"
    )
    ck_val_loss = ModelCheckpoint(monitor="val_loss", filename="model-{epoch:02d}-{val_loss:.2f}")
    ck_val_acc = ModelCheckpoint(
        monitor="val_acc", mode="max", filename="model-{epoch:02d}-{val_acc:.2f}"
    )

    data = MNISTDataModel()
    model = LeNet5ModernModel()
    logger = TensorBoardLogger(".", default_hp_metric=False, version="lenet_5_modern")
    trainer = pl.Trainer(
        max_epochs=1000,
        logger=logger,
        callbacks=[ck_train_loss, ck_train_acc, ck_val_loss, ck_val_acc],
    )
    tuner = Tuner(trainer)

    tuner.lr_find(model, data, min_lr=1e-3, max_lr=1e-1, num_training=1000)
    trainer.fit(model, data)
    trainer.test(model, data)

    trainer = pl.Trainer(logger=None)
    for ck in [ck_train_loss, ck_train_acc, ck_val_loss, ck_val_acc]:
        print(f"load from {ck}:")
        model_best = LeNet5Model.load_from_checkpoint(ck.best_model_path)
        trainer.test(model_best, data)


def test():
    for lr in (0.05, 0.5, 5):
        for optimizer in ("SGD", "Adam"):
            for activation in ("Tanh", "Sigmoid", "ReLU", "LeakyReLU"):
                for pool in ("avg", "max"):
                    for weight_decay in (0, 1e-5, 1e-4, 1e-3):
                        pl.seed_everything(42)
                        data = MNISTDataModel()
                        model = LeNet5Model(lr, weight_decay, optimizer, activation, pool)
                        logger = TensorBoardLogger(".", default_hp_metric=False)
                        trainer = pl.Trainer(max_epochs=100, logger=logger)
                        trainer.fit(model, data)
                        trainer.test(model, data)


if __name__ == "__main__":
    lenet_5_modern()
