#!/usr/bin/env python3

import os.path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.set_float32_matmul_precision("high")


def log_mse_loss(preds, labels):
    clipped_preds = torch.clamp(preds, 1, float("inf"))  # type: ignore
    mse = torch.sqrt(F.mse_loss(torch.log(clipped_preds), torch.log(labels)))  # type: ignore
    return mse


class KaggleDataModel(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        data_dir="../data",
        train_csv="kaggle_house_pred_train.csv",
        test_csv="kaggle_house_pred_test.csv",
        num_workers=8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_data_path = os.path.join(data_dir, train_csv)
        self.test_data_path = os.path.join(data_dir, test_csv)
        self.current_fold = 0
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # 读取数据
        train_data = pd.read_csv(self.train_data_path)
        test_data = pd.read_csv(self.test_data_path)

        # 标准化数据 x <- (x - μ) / σ
        all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
        numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
        all_features[numeric_features] = all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std())
        )
        all_features[numeric_features] = all_features[numeric_features].fillna(0)

        # 对离散值使用 One-Hot Encoding
        all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)

        # 构造数据
        n_train = train_data.shape[0]
        if stage == "fit" or stage is None:
            train_features = all_features[:n_train].values
            train_labels = train_data.SalePrice.values
            dataset = TensorDataset(
                torch.tensor(train_features, dtype=torch.float32),  # type: ignore
                torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1),  # type: ignore
            )
            self.train_data, self.val_data = random_split(dataset, [0.8, 0.2])
        if stage == "test" or stage is None:
            test_features = all_features[n_train:].values
            self.test_data = TensorDataset(torch.tensor(test_features, dtype=torch.float32))  # type: ignore

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


class KaggleModel(pl.LightningModule):
    def __init__(self, lr=1, weight_decay=1e-4, hidden_size=1024):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.input_size = 330
        self.output_size = 1

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def setup(self, stage=None):
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, self.output_size),
        )
        self.net.apply(self.init_weight)

    def forward(self, X):
        return self.net(X)

    def training_step(self, batch):
        features, labels = batch
        preds = self(features)
        loss = log_mse_loss(preds, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        features, labels = batch
        preds = self(features)
        loss = log_mse_loss(preds, labels)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch):
        features, labels = batch
        preds = self(features)
        loss = log_mse_loss(preds, labels)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        self.hparams["optim"] = "SGD"
        return optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def on_train_epoch_end(self):
        self.hparams["epoch"] = self.current_epoch
        self.logger.log_hyperparams(self.hparams)  # type: ignore


if __name__ == "__main__":
    data = KaggleDataModel(batch_size=256)
    model = KaggleModel(lr=10, weight_decay=1e-6, hidden_size=1024)
    trainer = pl.Trainer(max_epochs=400, log_every_n_steps=1)
    trainer.fit(model, data)


# Adam 的收敛速度要快一些，但是有可能收敛不到最小值，而且曲线波动较大
# SGD 的收敛速度稍慢（设置好参数后其实并不慢），收敛性很好
