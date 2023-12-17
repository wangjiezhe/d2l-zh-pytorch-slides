#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Resize, ToTensor

torch.set_float32_matmul_precision("medium")


class FashionMNISTDataModel(pl.LightningDataModule):
    def __init__(self, batch_size=128, data_dir="../data", num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.trans = Compose([ToTensor(), Resize(96, antialias=True)])

    def prepare_data(self):
        FashionMNIST(root=self.data_dir, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = FashionMNIST(root=self.data_dir, train=True, transform=self.trans)
            self.val_data = FashionMNIST(root=self.data_dir, train=False, transform=self.trans)
        elif stage == "test":
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


def init_cnn(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


class Residual(nn.Module):
    def __init__(self, output_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(output_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(output_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(output_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(output_channels))
    return blk


class Bottleneck(nn.Module):
    def __init__(self, middle_channels, output_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(middle_channels, kernel_size=1)
        self.conv2 = nn.LazyConv2d(middle_channels, kernel_size=3, padding=1, stride=strides)
        self.conv3 = nn.LazyConv2d(output_channels, kernel_size=1)
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(output_channels, kernel_size=1, stride=strides)
        else:
            self.conv4 = nn.LazyConv2d(output_channels, kernel_size=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        X = self.conv4(X)
        Y += X
        return F.relu(Y)


def bottleneck_block(bot_channels, output_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Bottleneck(bot_channels, output_channels, use_1x1conv=True, strides=2))
        elif i == 0:
            blk.append(Bottleneck(bot_channels, output_channels))
        else:
            blk.append(Bottleneck(bot_channels, output_channels))
    return blk


class ResNeXtBlock(nn.Module):
    def __init__(self, bot_channels, output_channels, groups, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(
            bot_channels, kernel_size=3, stride=strides, padding=1, groups=groups
        )
        self.conv3 = nn.LazyConv2d(output_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(output_channels, kernel_size=1, stride=strides)
        else:
            self.conv4 = nn.LazyConv2d(output_channels, kernel_size=1)
        self.bn4 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)


def resnext_block(bot_channels, output_channels, num_residuals, groups, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                ResNeXtBlock(bot_channels, output_channels, groups, use_1x1conv=True, strides=2)
            )
        elif i == 0:
            blk.append(ResNeXtBlock(bot_channels, output_channels, groups))
        else:
            blk.append(ResNeXtBlock(bot_channels, output_channels, groups))
    return blk


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=10, average="micro")
        self.val_acc = Accuracy(task="multiclass", num_classes=10, average="micro")
        self.test_acc = Accuracy(task="multiclass", num_classes=10, average="micro")

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


resnet_arch = {
    "18": ((64, 2), (128, 2), (256, 2), (512, 2)),
    "34": ((64, 3), (128, 4), (256, 6), (512, 3)),
    "50": ((64, 256, 3), (128, 512, 4), (256, 1024, 6), (512, 2048, 3)),
    "101": ((64, 256, 3), (128, 512, 4), (256, 1024, 23), (512, 2048, 3)),
    "152": ((64, 256, 3), (128, 512, 8), (256, 1024, 36), (512, 2048, 3)),
}

resnext_arch = {"50": ((128, 256, 3), (256, 512, 4), (512, 1024, 6), (1024, 2048, 3))}


class ResNet(Classifier):
    def __init__(self, block, arch, lr=0.001, momentum=0.9, weight_decay=1e-5, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add_module(
            "conv1",
            nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ),
        )
        for i in range(4):
            self.net.add_module(
                f"conv{i+2}", nn.Sequential(*block(*arch[i], first_block=(i == 0)))
            )
        self.net.add_module(
            "last",
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.LazyLinear(num_classes)),
        )
        self.inited = False

    def forward(self, x):
        return self.net(x)

    def setup(self, stage=None):
        if not self.inited:
            self.apply_init()

    def apply_init(self, dataloader=None):
        if dataloader is None:
            dataloader = self.trainer.datamodule.train_dataloader()
        dummy_batch = next(iter(dataloader))[0][0:1]
        self.forward(dummy_batch)
        self.net.apply(init_cnn)
        self.inited = True

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )


class ResNet18(ResNet):
    def __init__(self, lr=1e-3):
        super().__init__(resnet_block, resnet_arch["18"], lr)
        self.hparams["model"] = "ResNet18"


class ResNet34(ResNet):
    def __init__(self, lr=1e-3):
        super().__init__(resnet_block, resnet_arch["34"], lr)
        self.hparams["model"] = "ResNet34"


class ResNet50(ResNet):
    def __init__(self, lr=1e-3):
        super().__init__(bottleneck_block, resnet_arch["50"], lr)
        self.hparams["model"] = "ResNet50"


class ResNet101(ResNet):
    def __init__(self, lr=2e-4):
        super().__init__(bottleneck_block, resnet_arch["101"], lr)
        self.hparams["model"] = "ResNet101"


class ResNet152(ResNet):
    def __init__(self, lr=1e-6):
        super().__init__(bottleneck_block, resnet_arch["152"], lr)
        self.hparams["model"] = "ResNet152"


class ResNeXt50(ResNet):
    def __init__(self, lr=1e-3, groups=32):
        arch = [lt + (groups,) for lt in resnext_arch["50"]]
        super().__init__(resnext_block, arch, lr)
        self.hparams["model"] = "ResNeXt50"


def test_resnet18():
    data = FashionMNISTDataModel(batch_size=128)
    for lr in (0.01, 0.05, 0.1):
        for momentum in (0, 0.9):
            for weight_decay in (1e-3, 1e-4, 1e-5):
                model = ResNet(
                    resnet_block,
                    resnet_arch["18"],
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
                logger = TensorBoardLogger(".", default_hp_metric=False)
                trainer = pl.Trainer(max_epochs=10, logger=logger)
                trainer.fit(model, data)
    # Result：
    # highest train_acc: lr=0.01, momentum=0, weight_decay=1e-5 or 1e-4
    # highest val_acc: lr=0.05, momentum=0.9, weight_decay=1e-5 or 1e-4


def test2_resnet18():
    data = FashionMNISTDataModel(batch_size=128)
    lr = 0.0015
    for momentum in (0, 0.9):
        for weight_decay in (1e-3, 1e-4, 1e-5):
            model = ResNet(
                resnet_block,
                resnet_arch["18"],
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            logger = TensorBoardLogger(".", default_hp_metric=False)
            trainer = pl.Trainer(max_epochs=10, logger=logger)
            trainer.fit(model, data)
    # Result:
    # lr=0.0015, momentum=0.9, weight_decay=1e-5


def test3_resnet18():
    data = FashionMNISTDataModel(batch_size=128)
    for lr, momentum, weight_decay in (
        (0.01, 0, 1e-5),
        (0.05, 0.9, 1e-4),
        (0.0015, 0.9, 1e-5),
        (0.05, 0.9, 1e-5),
    ):
        model = ResNet(
            resnet_block,
            resnet_arch["18"],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        logger = TensorBoardLogger(".", default_hp_metric=False)
        trainer = pl.Trainer(max_epochs=20, logger=logger)
        trainer.fit(model, data)
    # Result:
    # (0.01, 0, 1e-5):      overfitting, train_loss = 1, val_loss = 0.922.
    # (0.05, 0.9, 1e-4):    overfitting, but train_loss is not always 1, val_loss fluctuates.
    # (0.0015, 0.9, 1e-5):  overfitting, train_loss = 1, val_loss = 0.924.
    # (0.05, 0.9, 1e-5):    overfitting, but train_loss is not always 1, val_loss fluctuates, but better than above.


def cks(name):
    ck_train_loss = ModelCheckpoint(
        dirpath="./checkpoints",
        monitor="train_loss",
        mode="min",
        filename=f"{name}" + "-{epoch:02d}-{train_loss:.2f}",
    )
    ck_train_acc = ModelCheckpoint(
        dirpath="./checkpoints",
        monitor="train_acc",
        mode="max",
        filename=f"{name}" + "-{epoch:02d}-{val_acc:.2f}",
    )
    ck_val_loss = ModelCheckpoint(
        dirpath="./checkpoints",
        monitor="val_loss",
        mode="min",
        filename=f"{name}" + "-{epoch:02d}-{val_loss:.2f}",
    )
    ck_val_acc = ModelCheckpoint(
        dirpath="./checkpoints",
        monitor="val_acc",
        mode="max",
        filename=f"{name}" + "-{epoch:02d}-{val_acc:.2f}",
    )
    return [ck_train_loss, ck_train_acc, ck_val_loss, ck_val_acc]


def train(model, epoch, logger_version=None, save_path=None, load_path=None, batch_size=128):
    data = FashionMNISTDataModel(batch_size=batch_size)
    if load_path is not None:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.inited = True
    logger = TensorBoardLogger(".", default_hp_metric=False, version=logger_version)
    trainer = pl.Trainer(max_epochs=epoch, logger=logger)
    trainer.fit(model, data)
    trainer.test(model, data)
    if save_path is not None:
        trainer.save_checkpoint(save_path)


def train_resnet18():
    train(ResNet18(), 20, "ResNet18", "ResNet18-epoch=20.ckpt")


def train_resnet34():
    train(ResNet34(), 20, "ResNet34", "ResNet34-epoch=20.ckpt")


def train_resnet34_from18():
    train(
        ResNet34(),
        20,
        "ResNet34-from18",
        "ResNet34-from18-epoch=20.ckpt",
        "ResNet18-epoch=20.ckpt",
    )


def train_resnet50():
    train(ResNet50(), 20, "ResNet50", "ResNet50-epoch=20.ckpt")


def train_resnet101():
    train(ResNet101(), 20, "ResNet101", "ResNet101-epoch=20.ckpt")


def train_resnet101_from50():
    train(
        ResNet101(lr=1e-4),
        20,
        "ResNet101-from50",
        "ResNet101-from50-epoch=20.ckpt",
        "ResNet50-epoch=20.ckpt",
    )


def train_resnet152():
    train(ResNet152(), 20, "ResNet152", "ResNet152-epoch=20.ckpt", batch_size=64)


def train_resnet152_from101():
    train(
        ResNet152(),
        20,
        "ResNet152-from101",
        "ResNet152-from101-epoch=20.ckpt",
        "ResNet101-epoch=20.ckpt",
        batch_size=64,
    )


def train_resnet152_from101_from50():
    train(
        ResNet152(),
        20,
        "ResNet152-from101-from50",
        "ResNet152-from101-from50-epoch=20.ckpt",
        "ResNet101-from50-epoch=20.ckpt",
        batch_size=64,
    )


def train_resnext50():
    train(ResNeXt50(lr=5e-3), 20, "ResNeXt50", "ResNeXt50-epoch=20.ckpt")


if __name__ == "__main__":
    # test_resnet18()
    # test2_resnet18()
    # test3_resnet18()
    # train_resnet18()
    # train_resnet34()
    # train_resnet34_from18()
    # train_resnet50()
    # train_resnet101()
    # train_resnet101_from50()
    # train_resnet152()
    train_resnext50()
    # train_resnet152_from101()
    # train_resnet152_from101_from50()
    # pass
