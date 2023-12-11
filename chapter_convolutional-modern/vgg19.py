#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Resize, ToTensor

torch.set_float32_matmul_precision("medium")

conv_arch = {
    "11": ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
    "13": ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512)),
    "16": ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)),
    "19": ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512)),
}

ratio = 4
small_conv_arch = {
    key: [(pair[0], pair[1] // ratio) for pair in value] for key, value in conv_arch.items()
}


class FashionMNISTDataModel(pl.LightningDataModule):
    def __init__(self, batch_size=128, data_dir="../data", num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.trans = Compose([ToTensor(), Resize(224, antialias=True)])

    def prepare_data(self):
        FashionMNIST(root=self.data_dir, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data, self.val_data = random_split(
                FashionMNIST(root=self.data_dir, train=True, transform=self.trans), [0.8, 0.2]
            )
        elif stage == "test" or stage is None:
            self.test_data = FashionMNIST(root=self.data_dir, train=False, transform=self.trans)

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


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class Vgg(pl.LightningModule):
    def __init__(self, conv_arch, lr=0.05, weight_decay=0):
        super().__init__()
        # self.lr = lr
        # self.weight_decay = weight_decay
        self.save_hyperparameters()

        conv_blks = []
        in_channels = 1
        for num_convs, out_channels in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        self.features = nn.Sequential(*conv_blks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        self.loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=10, average="micro")
        self.val_acc = Accuracy(task="multiclass", num_classes=10, average="micro")
        self.test_acc = Accuracy(task="multiclass", num_classes=10, average="micro")

    def forward(self, X):
        X = self.features(X)
        X = self.classifier(X)
        return X

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
        return loss

    def test_step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = self.loss(y_hat, y)
        acc = self.test_acc(y_hat, y)
        metrics = {"test_loss": loss, "test_acc": acc}
        self.log_dict(metrics, prog_bar=True)
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
        return torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )


def cks(name):
    ck_train_loss = ModelCheckpoint(
        monitor="train_loss", filename=f"{name}" + "-{epoch:02d}-{train_loss:.2f}"
    )
    ck_train_acc = ModelCheckpoint(
        monitor="train_acc", mode="max", filename=f"{name}" + "-{epoch:02d}-{val_acc:.2f}"
    )
    ck_val_loss = ModelCheckpoint(
        monitor="val_loss", filename=f"{name}" + "-{epoch:02d}-{val_loss:.2f}"
    )
    ck_val_acc = ModelCheckpoint(
        monitor="val_acc", mode="max", filename=f"{name}" + "-{epoch:02d}-{val_acc:.2f}"
    )
    return [ck_train_loss, ck_train_acc, ck_val_loss, ck_val_acc]


def main():
    pl.seed_everything(42)

    data = FashionMNISTDataModel(batch_size=128)

    model = Vgg(small_conv_arch["11"])
    logger = TensorBoardLogger(".", default_hp_metric=False)
    trainer = pl.Trainer(max_epochs=10, logger=logger, callbacks=cks("vgg11"))
    tuner = Tuner(trainer)
    tuner.lr_find(model, data, min_lr=1e-3, max_lr=1e-1, num_training=200)
    trainer.fit(model, data)
    trainer.test(model, data)
    trainer.save_checkpoint("vgg11_epoch=10.ckpt")

    # trainer = pl.Trainer(logger=None)
    # for ck in cks("vgg19"):
    #     print(f"load from {ck}:")
    #     model_best = Vgg(small_conv_arch["19"]).load_from_checkpoint(ck.best_model_path)
    #     trainer.test(model_best, data)


def load11to13():
    checkpoint = torch.load("vgg11_epoch=10.ckpt")
    data = FashionMNISTDataModel(batch_size=128)
    model = Vgg(small_conv_arch["13"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.hparams.lr = checkpoint["hyper_parameters"]["lr"]
    model.hparams.weight_decay = checkpoint["hyper_parameters"]["weight_decay"]
    logger = TensorBoardLogger(".", default_hp_metric=False)
    trainer = pl.Trainer(max_epochs=10, logger=logger, callbacks=cks("vgg13"))
    trainer.fit(model, data)
    trainer.test(model, data)
    trainer.save_checkpoint("vgg13_epoch=10.ckpt")


def load13():
    data = FashionMNISTDataModel(batch_size=128)
    model = Vgg.load_from_checkpoint("vgg13_epoch=10.ckpt")
    logger = TensorBoardLogger(".", default_hp_metric=False)
    trainer = pl.Trainer(max_epochs=10, logger=logger, callbacks=cks("vgg13"))
    trainer.fit(model, data)
    trainer.test(model, data)
    trainer.save_checkpoint("vgg13_epoch=20.ckpt")


if __name__ == "__main__":
    # main()
    load11to13()
    # load13()
