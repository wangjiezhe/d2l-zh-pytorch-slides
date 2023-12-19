#!/usr/bin/env python3


import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from resnet_lightning import Classifier, FashionMNISTDataModel, init_cnn, train
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Resize, ToTensor

torch.set_float32_matmul_precision("medium")


def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(), nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
    )


class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super().__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )


densenet_arch = {
    "121": (32, (6, 12, 24, 16)),
    "169": (32, (6, 12, 32, 32)),
    "201": (32, (6, 12, 48, 32)),
    "161": (64, (6, 12, 36, 24)),
}


class DenseNet(Classifier):
    def b1(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def __init__(
        self,
        num_channels=64,
        growth_rate=32,
        arch=(4, 4, 4, 4),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-5,
        num_classes=10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b1(num_channels))
        for i, num_convs in enumerate(arch):
            self.net.add_module(f"dense_blk{i+1}", DenseBlock(num_convs, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(f"tran_blk{i+1}", transition_block(num_channels))
        self.net.add_module(
            "last",
            nn.Sequential(
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.LazyLinear(num_classes),
            ),
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


class DenseNet121(DenseNet):
    def __init__(self, lr=0.1):
        super().__init__(growth_rate=densenet_arch["121"][0], arch=densenet_arch["121"][1], lr=lr)


class DenseNet169(DenseNet):
    def __init__(self, lr=0.1):
        super().__init__(growth_rate=densenet_arch["169"][0], arch=densenet_arch["169"][1], lr=lr)


class DenseNet201(DenseNet):
    def __init__(self, lr=0.1):
        super().__init__(growth_rate=densenet_arch["201"][0], arch=densenet_arch["201"][1], lr=lr)


class DenseNet161(DenseNet):
    def __init__(self, lr=0.1):
        super().__init__(growth_rate=densenet_arch["161"][0], arch=densenet_arch["161"][1], lr=lr)


def find_lr(model, min_lr, max_lr, num_training, batch_size=128):
    data = FashionMNISTDataModel(batch_size=batch_size)
    trainer = pl.Trainer(max_epochs=20)
    tuner = Tuner(trainer)
    return tuner.lr_find(model, data, min_lr=min_lr, max_lr=max_lr, num_training=num_training)


if __name__ == "__main__":
    train(DenseNet121(), 20, "DenseNet121", "DenseNet121-epoch=20.ckpt", batch_size=64)
    train(DenseNet169(), 20, "DenseNet169", "DenseNet169-epoch=20.ckpt", batch_size=32)
    train(DenseNet201(), 20, "DenseNet201", "DenseNet201-epoch=20.ckpt", batch_size=32)
    train(DenseNet161(), 20, "DenseNet161", "DenseNet161-epoch=20.ckpt", batch_size=32)
    pass
