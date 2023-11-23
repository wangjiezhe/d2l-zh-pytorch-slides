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
num_epochs = 20
gpu = torch.device('cuda')

## 导入MNIST数据
trans = torchvision.transforms.ToTensor()

def load_mnist(batch_size):
  mnist_train = torchvision.datasets.MNIST(root="../data", train=True, transform=trans, download=True)
  mnist_test = torchvision.datasets.MNIST(root="../data", train=False, transform=trans, download=True)
  train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True)
  test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False)
  return train_iter, test_iter

def load_fashionmnist(batch_size):
  mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
  mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
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
      X, y = X.to(gpu), y.to(gpu)
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
    X, y = X.to(gpu), y.to(gpu)
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

def main(load):
  train_iter, test_iter = load(batch_size)

  net = nn.Sequential(nn.Flatten(),
                      nn.Linear(num_inputs, num_outputs))
  net = torch.jit.script(net.to(gpu))
  net.apply(init_weights)

  loss = nn.CrossEntropyLoss()
  loss = torch.jit.script(loss.to(gpu))

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
  for X, y in train_iter: break
  show_images(X.reshape(fig_size, 28, 28), num_rows, num_cols, titles=list(y.numpy()))

def show_predict(net, test_iter, n=9):
  with torch.no_grad():
    for X, y in test_iter: break
    net = net.to('cpu')
    trues = list(y.numpy())
    preds = list(net(X).argmax(axis=1).numpy())
    titles = [f'{true}\n{pred}' for true, pred in zip(trues, preds)]
    show_images(X[0:n], 1, n, titles=titles[0:n])


if __name__ == '__main__':
  # show_mnist(5,9)
  print("MNIST:")
  main(load_mnist)
  print("\nFashionMNIST:")
  main(load_fashionmnist)


### Output

# MNIST:
# epoch 1, accuracy 0.885400
# epoch 2, accuracy 0.899600
# epoch 3, accuracy 0.904600
# epoch 4, accuracy 0.908900
# epoch 5, accuracy 0.910900
# epoch 6, accuracy 0.913300
# epoch 7, accuracy 0.915500
# epoch 8, accuracy 0.915500
# epoch 9, accuracy 0.916600
# epoch 10, accuracy 0.917900
# epoch 11, accuracy 0.915900
# epoch 12, accuracy 0.918700
# epoch 13, accuracy 0.917500
# epoch 14, accuracy 0.919700
# epoch 15, accuracy 0.920400
# epoch 16, accuracy 0.920000
# epoch 17, accuracy 0.920500
# epoch 18, accuracy 0.920200
# epoch 19, accuracy 0.920900
# epoch 20, accuracy 0.920900

# FashionMNIST:
# epoch 1, accuracy 0.793900
# epoch 2, accuracy 0.801300
# epoch 3, accuracy 0.817300
# epoch 4, accuracy 0.816000
# epoch 5, accuracy 0.817200
# epoch 6, accuracy 0.823200
# epoch 7, accuracy 0.827700
# epoch 8, accuracy 0.817200
# epoch 9, accuracy 0.833400
# epoch 10, accuracy 0.832300
# epoch 11, accuracy 0.832300
# epoch 12, accuracy 0.834300
# epoch 13, accuracy 0.829400
# epoch 14, accuracy 0.829300
# epoch 15, accuracy 0.832900
# epoch 16, accuracy 0.833100
# epoch 17, accuracy 0.838100
# epoch 18, accuracy 0.830000
# epoch 19, accuracy 0.837600
# epoch 20, accuracy 0.833900
