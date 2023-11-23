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
num_hiddens = 1024
batch_size = 256
learning_rate = 0.5
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

@torch.no_grad()
def evaluate_accuracy(net, data_iter):
  net.eval()
  num_correct = 0
  num_total = 0
  for X, y in data_iter:
    X, y = X.to(gpu), y.to(gpu)
    num_correct += accuracy(net(X), y)
    num_total += y.numel()
  return num_correct / num_total

## 参数初始化
@torch.no_grad
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

## 数据集可视化
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

@torch.no_grad()
def show_predict(net, test_iter, n=9):
  for X, y in test_iter: break
  net = net.to('cpu')
  trues = list(y.numpy())
  preds = list(net(X).argmax(dim=1).numpy())
  titles = [f'{true}\n{pred}' for true, pred in zip(trues, preds)]
  show_images(X[0:n].reshape(n,28,28), 1, n, titles=titles[0:n])


if __name__ == '__main__':
  # show_mnist(5,9)

  print("MNIST:")
  train_iter, test_iter = load_mnist(batch_size)
  net = nn.Sequential(nn.Flatten(),
                      nn.Linear(num_inputs, num_hiddens),
                      nn.ReLU(),
                      nn.Linear(num_hiddens, num_outputs))
  net = torch.jit.script(net.to(gpu))
  net.apply(init_weights)
  loss = nn.CrossEntropyLoss()
  loss = torch.jit.script(loss.to(gpu))
  trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)
  train(net, train_iter, test_iter, loss, num_epochs, trainer)

  # show_predict(net, test_iter)

  print("\nFashionMNIST:")
  train_iter, test_iter = load_fashionmnist(batch_size)
  net = nn.Sequential(nn.Flatten(),
                      nn.Linear(num_inputs, num_hiddens),
                      nn.ReLU(),
                      nn.Linear(num_hiddens, num_outputs))
  net = torch.jit.script(net.to(gpu))
  net.apply(init_weights)
  loss = nn.CrossEntropyLoss()
  loss = torch.jit.script(loss.to(gpu))
  trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)
  train(net, train_iter, test_iter, loss, num_epochs, trainer)


### Output

# MNIST:
# epoch 1, accuracy 0.935200
# epoch 2, accuracy 0.958200
# epoch 3, accuracy 0.968100
# epoch 4, accuracy 0.971600
# epoch 5, accuracy 0.970600
# epoch 6, accuracy 0.975200
# epoch 7, accuracy 0.962800
# epoch 8, accuracy 0.978800
# epoch 9, accuracy 0.978900
# epoch 10, accuracy 0.980100
# epoch 11, accuracy 0.981800
# epoch 12, accuracy 0.980600
# epoch 13, accuracy 0.978800
# epoch 14, accuracy 0.980300
# epoch 15, accuracy 0.980900
# epoch 16, accuracy 0.981400
# epoch 17, accuracy 0.981700
# epoch 18, accuracy 0.977100
# epoch 19, accuracy 0.982600
# epoch 20, accuracy 0.981600

# FashionMNIST:
# epoch 1, accuracy 0.785100
# epoch 2, accuracy 0.768600
# epoch 3, accuracy 0.823200
# epoch 4, accuracy 0.798400
# epoch 5, accuracy 0.857400
# epoch 6, accuracy 0.809800
# epoch 7, accuracy 0.852500
# epoch 8, accuracy 0.861900
# epoch 9, accuracy 0.848000
# epoch 10, accuracy 0.869300
# epoch 11, accuracy 0.867200
# epoch 12, accuracy 0.882400
# epoch 13, accuracy 0.871100
# epoch 14, accuracy 0.869300
# epoch 15, accuracy 0.877800
# epoch 16, accuracy 0.854100
# epoch 17, accuracy 0.874600
# epoch 18, accuracy 0.845000
# epoch 19, accuracy 0.881800
# epoch 20, accuracy 0.877100
