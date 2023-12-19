#!/usr/bin/env python3

import os.path
import time

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchmetrics.regression import MeanSquaredLogError

torch.set_float32_matmul_precision("medium")


def init_cnn(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def log_mse_loss(preds, labels):
    clipped_preds = torch.clamp(preds, 1, float("inf"))
    mse = torch.sqrt(F.mse_loss(torch.log(clipped_preds), torch.log(labels)))
    return mse


def train(
    model,
    epoch,
    logger_version=None,
    save_path=None,
    load_path=None,
    batch_size=256,
    test=True,
):
    data = KaggleData(batch_size=batch_size)
    if load_path is not None:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.inited = True
    logger = TensorBoardLogger(".", name="kaggle_logs", default_hp_metric=False, version=logger_version)
    trainer = pl.Trainer(max_epochs=epoch, logger=logger, log_every_n_steps=5)
    trainer.fit(model, data)
    if test:
        trainer.test(model, data)
    if save_path is not None:
        trainer.save_checkpoint(save_path)


class KaggleData(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        data_dir="../data",
        train_csv="kaggle_house_pred_train.csv",
        test_csv="kaggle_house_pred_test.csv",
        num_workers=8,
    ):
        super().__init__()
        self.train_data_path = os.path.join(data_dir, train_csv)
        self.test_data_path = os.path.join(data_dir, test_csv)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # 读取数据
        self.origin_train_data = pd.read_csv(self.train_data_path)
        self.origin_test_data = pd.read_csv(self.test_data_path)
        self.test_data_id = self.origin_test_data["Id"]

        # 标准化数据 x <- (x - μ) / σ
        all_features = pd.concat((self.origin_train_data.iloc[:, 1:-1], self.origin_test_data.iloc[:, 1:]))
        numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
        all_features[numeric_features] = all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std())
        )
        all_features[numeric_features] = all_features[numeric_features].fillna(0)

        # 对离散值使用 One-Hot Encoding
        self.all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)

    def setup(self, stage=None):
        # 构造数据
        n_train = self.origin_train_data.shape[0]
        train_features = self.all_features[:n_train].values
        train_labels = self.origin_train_data.SalePrice.values
        if stage == "fit" or stage is None:
            self.train_data, self.val_data = random_split(
                TensorDataset(
                    torch.tensor(train_features, dtype=torch.float32),
                    torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1),
                ),
                [0.8, 0.2],
            )
        if stage == "test" or stage is None:
            self.test_data = TensorDataset(
                torch.tensor(train_features, dtype=torch.float32),
                torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1),
            )
        if stage == "predict" or stage is None:
            predict_features = self.all_features[n_train:].values
            self.predict_data = TensorDataset(torch.tensor(predict_features, dtype=torch.float32))

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.train_loss = MeanSquaredLogError()
        self.val_loss = MeanSquaredLogError()
        self.test_loss = MeanSquaredLogError()

    def training_step(self, batch):
        features, labels = batch
        preds = self(features)
        clipped_preds = torch.clamp(preds, 0, float("inf"))
        loss = torch.sqrt(self.train_loss(clipped_preds, labels))
        # loss = log_mse_loss(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        features, labels = batch
        preds = self(features)
        clipped_preds = torch.clamp(preds, 0, float("inf"))
        loss = torch.sqrt(self.val_loss(clipped_preds, labels))
        # loss = log_mse_loss(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch):
        features, labels = batch
        preds = self(features)
        clipped_preds = torch.clamp(preds, 0, float("inf"))
        loss = torch.sqrt(self.test_loss(clipped_preds, labels))
        # loss = log_mse_loss(preds, labels)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch):
        pass

    def on_predict_end(self):
        features = self.trainer.datamodule.predict_data.tensors[0]
        preds = self.net(features).detach().numpy()
        submission = pd.concat(
            [self.trainer.datamodule.test_data_id, pd.Series(preds.reshape(1, -1)[0])],
            axis=1,
            keys=("Id", "SalePrice"),
        )
        submission.to_csv("submission.csv", index=False)

    def on_test_end(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/train_loss": torch.sqrt(self.train_loss.compute()),
                "hp/val_loss": torch.sqrt(self.val_loss.compute()),
                "hp/test_loss": torch.sqrt(self.test_loss.compute()),
            },
        )


class DenseMLPBlock(nn.Module):
    def __init__(self, input_size, feature_size, layers, dropout=0.5):
        super().__init__()
        blk = []
        for i in range(layers):
            blk.append(
                nn.Sequential(
                    nn.BatchNorm1d(input_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(input_size, feature_size),
                )
            )
            input_size = feature_size
        self.net = nn.Sequential(*blk)

    def forward(self, x):
        y = self.net(x)
        return torch.cat((x, y), dim=1)


class DenseMLP(Classifier):
    def __init__(self, arch=((64, 2), (128, 2), (256, 2), (512, 2)), lr=0.1, momentum=0.9, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.inited = False
        blk = []
        input_size = 330
        for feature_size, layers in arch:
            blk.append(DenseMLPBlock(input_size, feature_size, layers))
            input_size += feature_size
            blk.append(
                nn.Sequential(
                    nn.BatchNorm1d(input_size),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(input_size, input_size // 2),
                )
            )
            input_size //= 2
        self.net = nn.Sequential(*blk, nn.Linear(input_size, 1))

    def forward(self, x):
        return self.net(x)

    def setup(self, stage=None):
        if not self.inited:
            self.apply_init()

    def apply_init(self, dataloader=None):
        if dataloader is None:
            dataloader = self.trainer.datamodule.train_dataloader()
        dummy_batch = next(iter(dataloader))[0][0:2]
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


# def conv_block(num_channels):
#     return nn.Sequential(
#         nn.LazyBatchNorm1d(), nn.ReLU(), nn.LazyConv1d(num_channels, kernel_size=3, padding=1)
#     )


# class DenseBlock(nn.Module):
#     def __init__(self, num_convs, num_channels):
#         super().__init__()
#         layer = []
#         for i in range(num_convs):
#             layer.append(conv_block(num_channels))
#         self.net = nn.Sequential(*layer)

#     def forward(self, X):
#         for blk in self.net:
#             Y = blk(X)
#             X = torch.cat((X, Y), dim=1)
#         return X


# def transition_block(num_channels):
#     return nn.Sequential(
#         nn.LazyBatchNorm1d(),
#         nn.ReLU(),
#         nn.LazyConv1d(num_channels, kernel_size=1),
#         nn.AvgPool1d(kernel_size=2, stride=2),
#     )


# densenet_arch = {
#     "121": (32, (6, 12, 24, 16)),
#     "169": (32, (6, 12, 32, 32)),
#     "201": (32, (6, 12, 48, 32)),
#     "161": (64, (6, 12, 36, 24)),
# }


# class DenseNet(Classifier):
#     def b1(self, num_channels):
#         return nn.Sequential(
#             nn.LazyConv1d(num_channels, kernel_size=7, stride=2, padding=3),
#             nn.LazyBatchNorm1d(),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
#         )

#     def __init__(
#         self,
#         num_channels=64,
#         growth_rate=32,
#         arch=(4, 4, 4, 4),
#         lr=0.1,
#         momentum=0.9,
#         weight_decay=1e-5,
#         num_classes=1,
#     ):
#         super().__init__()
#         self.save_hyperparameters()
#         self.net = nn.Sequential(self.b1(num_channels))
#         for i, num_convs in enumerate(arch):
#             self.net.add_module(f"dense_blk{i+1}", DenseBlock(num_convs, growth_rate))
#             num_channels += num_convs * growth_rate
#             if i != len(arch) - 1:
#                 num_channels //= 2
#                 self.net.add_module(f"tran_blk{i+1}", transition_block(num_channels))
#         self.net.add_module(
#             "last",
#             nn.Sequential(
#                 nn.LazyBatchNorm1d(),
#                 nn.ReLU(),
#                 nn.AdaptiveAvgPool1d(1),
#                 nn.Flatten(),
#                 nn.LazyLinear(num_classes),
#             ),
#         )
#         self.inited = False

#     def forward(self, x):
#         return self.net(x)

#     def setup(self, stage=None):
#         if not self.inited:
#             self.apply_init()

#     def apply_init(self, dataloader=None):
#         if dataloader is None:
#             dataloader = self.trainer.datamodule.train_dataloader()
#         dummy_batch = next(iter(dataloader))[0][0:1]
#         self.forward(dummy_batch)
#         self.net.apply(init_cnn)
#         self.inited = True

#     def configure_optimizers(self):
#         return torch.optim.SGD(
#             self.parameters(),
#             lr=self.hparams.lr,
#             momentum=self.hparams.momentum,
#             weight_decay=self.hparams.weight_decay,
#         )


# class DenseNet121(DenseNet):
#     def __init__(self, lr=0.1):
#         super().__init__(growth_rate=densenet_arch["121"][0], arch=densenet_arch["121"][1], lr=lr)


# class DenseNet169(DenseNet):
#     def __init__(self, lr=0.1):
#         super().__init__(growth_rate=densenet_arch["169"][0], arch=densenet_arch["169"][1], lr=lr)


# class DenseNet201(DenseNet):
#     def __init__(self, lr=0.1):
#         super().__init__(growth_rate=densenet_arch["201"][0], arch=densenet_arch["201"][1], lr=lr)


# class DenseNet161(DenseNet):
#     def __init__(self, lr=0.1):
#         super().__init__(growth_rate=densenet_arch["161"][0], arch=densenet_arch["161"][1], lr=lr)


# class KaggleData_DN(KaggleData):
#     def __init__(
#         self,
#         batch_size=64,
#         data_dir="../data",
#         train_csv="kaggle_house_pred_train.csv",
#         test_csv="kaggle_house_pred_test.csv",
#         num_workers=8,
#     ):
#         super().__init__(batch_size, data_dir, train_csv, test_csv, num_workers)

#     def setup(self, stage=None):
#         n_train = self.origin_train_data.shape[0]
#         train_features = self.all_features[:n_train].values
#         train_labels = self.origin_train_data.SalePrice.values
#         if stage == "fit" or stage is None:
#             self.train_data, self.val_data = random_split(
#                 TensorDataset(
#                     torch.tensor(train_features, dtype=torch.float32).unsqueeze(dim=1),
#                     torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1),
#                 ),
#                 [0.8, 0.2],
#             )
#         if stage == "test" or stage is None:
#             self.test_data = TensorDataset(
#                 torch.tensor(train_features, dtype=torch.float32).unsqueeze(dim=1),
#                 torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1),
#             )
#         if stage == "predict" or stage is None:
#             predict_features = self.all_features[n_train:].values
#             self.predict_data = TensorDataset(
#                 torch.tensor(predict_features, dtype=torch.float32).unsqueeze(dim=1)
#             )


# def train_dn(
#     model,
#     epoch,
#     logger_version=None,
#     save_path=None,
#     load_path=None,
#     batch_size=256,
#     test=True,
# ):
#     data = KaggleData_DN(batch_size=batch_size, num_workers=16)
#     if load_path is not None:
#         checkpoint = torch.load(load_path)
#         model.load_state_dict(checkpoint["state_dict"], strict=False)
#         model.inited = True
#     logger = TensorBoardLogger(".", name="kaggle_logs", default_hp_metric=False, version=logger_version)
#     trainer = pl.Trainer(max_epochs=epoch, logger=logger, log_every_n_steps=5)
#     trainer.fit(model, data)
#     if test:
#         trainer.test(model, data)
#     if save_path is not None:
#         trainer.save_checkpoint(save_path)


# class KaggleData_KFold(KaggleData):
#     def __init__(self, num_folds=5, batch_size=64):
#         super().__init__(batch_size=batch_size)
#         self.num_folds = num_folds
#         self.current_fold = 0

#     def setup(self, stage=None):
#         n_train = self.origin_train_data.shape[0]
#         features = self.all_features[:n_train]
#         labels = self.origin_train_data.SalePrice
#         kf = KFold(self.num_folds)
#         train_indices, val_indices = list(kf.split(features))[self.current_fold]
#         train_features = features.iloc[train_indices].values
#         train_labels = labels.iloc[train_indices].values
#         val_features = features.iloc[val_indices].values
#         val_labels = labels.iloc[val_indices].values

#         if stage == "fit" or stage is None:
#             self.train_data = TensorDataset(
#                 torch.tensor(train_features, dtype=torch.float32),
#                 torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1),
#             )
#             self.val_data = TensorDataset(
#                 torch.tensor(val_features, dtype=torch.float32),
#                 torch.tensor(val_labels, dtype=torch.float32).reshape(-1, 1),
#             )
#         if stage == "test" or stage is None:
#             self.test_data = TensorDataset(
#                 torch.tensor(val_features, dtype=torch.float32),
#                 torch.tensor(val_labels, dtype=torch.float32).reshape(-1, 1),
#             )


# def train_kfold(model, epoch, logger_version=None, load_path=None, batch_size=256):
#     data = KaggleData_KFold(batch_size=batch_size)
#     if load_path is not None:
#         checkpoint = torch.load(load_path)
#         model.load_state_dict(checkpoint["state_dict"], strict=False)
#         model.inited = True
#     for i in range(data.num_folds):
#         data.current_fold = i
#         logger = TensorBoardLogger(".", name="kaggle_logs", default_hp_metric=False, version=logger_version)
#         trainer = pl.Trainer(max_epochs=epoch, logger=logger, log_every_n_steps=6)
#         trainer.fit(model, data)
