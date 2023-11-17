#!/usr/bin/env python3

import torch
import torchvision

from torch import nn

from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

backend_inline.set_matplotlib_formats('svg')

## 超参数
num_inputs = 28*28
num_outputs = 10
batch_size = 256
learning_rate = 0.1
num_epochs = 10

## 导入MNIST数据
trans = torchvision.transforms.ToTensor()

def load_mnist(batch_size):
  mnist_train = torchvision.datasets.MNIST(root="../data", train=True, transform=trans, download=True)
  mnist_test = torchvision.datasets.MNIST(root="../data", train=False, transform=trans, download=True)
  train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True)
  test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False)
  return train_iter, test_iter

## 准确度
def accuracy(y_hat, y):
  y_hat = y_hat.argmax(dim=1)
  cmp = y_hat.type(y.dtype) == y
  return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
  net.eval()
  num_correct = 0
  num_total = 0
  with torch.no_grad():
    for X, y in data_iter:
      num_correct += accuracy(net(X), y)
      num_total += y.numel()
  return num_correct / num_total

## 参数初始化
def init_weights(m):
  if type(m) == nn.Linear:
    nn.init.normal_(m.weight,std=0.01)

## 训练（一个迭代周期）
def train_epoch(net, train_iter, loss, updater):
  net.train()
  for X, y in train_iter:
    y_hat = net(X)
    l = loss(y_hat, y)
    updater.zero_grad()
    l.backward()
    updater.step()

## 训练模型
def train(net, train_iter, test_iter, loss, num_epochs, updater):
  for epoch in range(num_epochs):
    train_epoch(net, train_iter, loss, updater)
    acc = evaluate_accuracy(net, test_iter)
    print(f'epoch {epoch + 1}, accuracy {acc:f}')
    
def main():
  train_iter, test_iter = load_mnist(batch_size)

  net = nn.Sequential(nn.Flatten(),
                      nn.Linear(num_inputs, num_outputs))
  net.apply(init_weights)

  loss = nn.CrossEntropyLoss()

  trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)

  train(net, train_iter, test_iter, loss, num_epochs, trainer)

## 可视化
def show_images(imgs, num_rows, num_cols, titles=None):
  _, axes = plt.subplots(num_rows, num_cols)
  axes = axes.flatten()
  for i, (ax, img) in enumerate(zip(axes, imgs)):
    ax.imshow(img)
    ax.axis('off')
    if titles:
      ax.set_title(titles[i])
  return axes

def show_mnist(num_rows, num_cols):
  fig_size = num_rows * num_cols
  train_iter, _ = load_mnist(batch_size=fig_size)
  for X, y in train_iter:
    break
  show_images(X.reshape(fig_size, 28, 28), num_rows, num_cols, titles=list(y.numpy()))
  
def show_predict(net, test_iter, n=9):
  for X, y in test_iter: break
  trues = list(y.numpy())
  preds = list(net(X).argmax(axis=1).numpy())
  titles = [f'{true}\n{pred}' for true, pred in zip(trues, preds)]
  show_images(X[0:n], 1, n, titles=titles[0:n])
  

if __name__ == '__main__':
  show_mnist(5,9)
  main()