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
weight_decay = 0
num_epochs = 50
num_workers = 8
gpu = torch.device('cuda')

## 导入MNIST数据
trans = torchvision.transforms.ToTensor()

def load_mnist(batch_size):
  mnist_train = torchvision.datasets.MNIST(root="../data", train=True, transform=trans, download=True)
  mnist_test = torchvision.datasets.MNIST(root="../data", train=False, transform=trans, download=True)
  train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
  test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
  return train_iter, test_iter

def load_fashionmnist(batch_size):
  mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
  mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
  train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
  test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
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
    nn.init.kaiming_normal_(nonlinearity='relu')

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

  print("MNIST (1 hidden layer with 1024 nodes):")
  train_iter, test_iter = load_mnist(batch_size)
  net = nn.Sequential(nn.Flatten(),
                      nn.Linear(num_inputs, num_hiddens),
                      nn.ReLU(),
                      nn.Linear(num_hiddens, num_outputs))
  net = torch.jit.script(net.to(gpu))
  net.apply(init_weights)
  loss = nn.CrossEntropyLoss()
  loss = torch.jit.script(loss.to(gpu))
  trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)
  train(net, train_iter, test_iter, loss, num_epochs, trainer)

  print("\nMNIST (2 hidden layer with 512/64 nodes):")
  train_iter, test_iter = load_mnist(batch_size)
  net = nn.Sequential(nn.Flatten(),
                      nn.Linear(num_inputs, 512), nn.ReLU(),
                      nn.Linear(512, 64), nn.ReLU(),
                      nn.Linear(64, num_outputs))
  net = torch.jit.script(net.to(gpu))
  net.apply(init_weights)
  loss = nn.CrossEntropyLoss()
  loss = torch.jit.script(loss.to(gpu))
  trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)
  train(net, train_iter, test_iter, loss, num_epochs, trainer)

  print("MNIST (1 hidden layer with 2048 nodes (dropout 0.5)):")
  train_iter, test_iter = load_mnist(batch_size)
  net = nn.Sequential(nn.Flatten(),
                      nn.Linear(num_inputs, num_hiddens),
                      nn.ReLU(),
                      nn.Linear(num_hiddens, num_outputs))
  net = torch.jit.script(net.to(gpu))
  net.apply(init_weights)
  loss = nn.CrossEntropyLoss()
  loss = torch.jit.script(loss.to(gpu))
  trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)
  train(net, train_iter, test_iter, loss, num_epochs, trainer)

  # show_predict(net, test_iter)

  print("\nFashionMNIST (1 hidden layer with 1024 nodes):")
  train_iter, test_iter = load_fashionmnist(batch_size)
  net = nn.Sequential(nn.Flatten(),
                      nn.Linear(num_inputs, num_hiddens),
                      nn.ReLU(),
                      nn.Linear(num_hiddens, num_outputs))
  net = torch.jit.script(net.to(gpu))
  net.apply(init_weights)
  loss = nn.CrossEntropyLoss()
  loss = torch.jit.script(loss.to(gpu))
  trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)
  train(net, train_iter, test_iter, loss, num_epochs, trainer)

  print("\nFashionMNIST (2 hidden layer with 512/64 nodes):")
  train_iter, test_iter = load_fashionmnist(batch_size)
  net = nn.Sequential(nn.Flatten(),
                      nn.Linear(num_inputs, 512), nn.ReLU(),
                      nn.Linear(512, 64), nn.ReLU(),
                      nn.Linear(64, num_outputs))
  net = torch.jit.script(net.to(gpu))
  net.apply(init_weights)
  loss = nn.CrossEntropyLoss()
  loss = torch.jit.script(loss.to(gpu))
  trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)
  train(net, train_iter, test_iter, loss, num_epochs, trainer)


  print("\nFashionMNIST (2 hidden layer with 512/64 nodes and dropout):")
  train_iter, test_iter = load_fashionmnist(batch_size)
  net = nn.Sequential(nn.Flatten(),
                      nn.Linear(num_inputs, 512), nn.ReLU(), nn.Dropout(0.2),
                      nn.Linear(512, 64), nn.ReLU(), nn.Dropout(),
                      nn.Linear(64, num_outputs))
  net = torch.jit.script(net.to(gpu))
  net.apply(init_weights)
  loss = nn.CrossEntropyLoss()
  loss = torch.jit.script(loss.to(gpu))
  trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)
  train(net, train_iter, test_iter, loss, num_epochs, trainer)


### Output

# MNIST (1 hidden layer with 1024 nodes):
# epoch 1, accuracy 0.928400
# epoch 2, accuracy 0.955300
# epoch 3, accuracy 0.967800
# epoch 4, accuracy 0.964800
# epoch 5, accuracy 0.972800
# epoch 6, accuracy 0.977200
# epoch 7, accuracy 0.975900
# epoch 8, accuracy 0.978400
# epoch 9, accuracy 0.980300
# epoch 10, accuracy 0.980400
# epoch 11, accuracy 0.981200
# epoch 12, accuracy 0.980400
# epoch 13, accuracy 0.980100
# epoch 14, accuracy 0.981700
# epoch 15, accuracy 0.981300
# epoch 16, accuracy 0.981500
# epoch 17, accuracy 0.981400
# epoch 18, accuracy 0.982300
# epoch 19, accuracy 0.980800
# epoch 20, accuracy 0.981800
# epoch 21, accuracy 0.982000
# epoch 22, accuracy 0.981700
# epoch 23, accuracy 0.982500
# epoch 24, accuracy 0.982400
# epoch 25, accuracy 0.981200
# epoch 26, accuracy 0.982100
# epoch 27, accuracy 0.982500
# epoch 28, accuracy 0.982500
# epoch 29, accuracy 0.982200
# epoch 30, accuracy 0.982600
# epoch 31, accuracy 0.982700
# epoch 32, accuracy 0.982100
# epoch 33, accuracy 0.982800
# epoch 34, accuracy 0.982300
# epoch 35, accuracy 0.982200
# epoch 36, accuracy 0.982800
# epoch 37, accuracy 0.982600
# epoch 38, accuracy 0.982600
# epoch 39, accuracy 0.982400
# epoch 40, accuracy 0.982800
# epoch 41, accuracy 0.982700
# epoch 42, accuracy 0.982600
# epoch 43, accuracy 0.982900
# epoch 44, accuracy 0.982900
# epoch 45, accuracy 0.982900
# epoch 46, accuracy 0.982900
# epoch 47, accuracy 0.983300
# epoch 48, accuracy 0.983300
# epoch 49, accuracy 0.982300
# epoch 50, accuracy 0.983600

# MNIST (2 hidden layer with 512/64 nodes):
# epoch 1, accuracy 0.898300
# epoch 2, accuracy 0.960500
# epoch 3, accuracy 0.969800
# epoch 4, accuracy 0.973900
# epoch 5, accuracy 0.978000
# epoch 6, accuracy 0.973700
# epoch 7, accuracy 0.981500
# epoch 8, accuracy 0.980600
# epoch 9, accuracy 0.982600
# epoch 10, accuracy 0.983000
# epoch 11, accuracy 0.971200
# epoch 12, accuracy 0.980300
# epoch 13, accuracy 0.983500
# epoch 14, accuracy 0.982300
# epoch 15, accuracy 0.984200
# epoch 16, accuracy 0.983500
# epoch 17, accuracy 0.984300
# epoch 18, accuracy 0.984100
# epoch 19, accuracy 0.983800
# epoch 20, accuracy 0.984300
# epoch 21, accuracy 0.983400
# epoch 22, accuracy 0.984100
# epoch 23, accuracy 0.982900
# epoch 24, accuracy 0.983900
# epoch 25, accuracy 0.983700
# epoch 26, accuracy 0.983700
# epoch 27, accuracy 0.984000
# epoch 28, accuracy 0.983900
# epoch 29, accuracy 0.983700
# epoch 30, accuracy 0.984200
# epoch 31, accuracy 0.984200
# epoch 32, accuracy 0.984400
# epoch 33, accuracy 0.984000
# epoch 34, accuracy 0.984100
# epoch 35, accuracy 0.984200
# epoch 36, accuracy 0.984200
# epoch 37, accuracy 0.984300
# epoch 38, accuracy 0.984300
# epoch 39, accuracy 0.984100
# epoch 40, accuracy 0.984200
# epoch 41, accuracy 0.984400
# epoch 42, accuracy 0.984300
# epoch 43, accuracy 0.984300
# epoch 44, accuracy 0.984200
# epoch 45, accuracy 0.984300
# epoch 46, accuracy 0.984000
# epoch 47, accuracy 0.984200
# epoch 48, accuracy 0.984100
# epoch 49, accuracy 0.984300
# epoch 50, accuracy 0.984400

# MNIST (1 hidden layer with 2048 nodes (dropout 0.5)):
# epoch 1, accuracy 0.943600
# epoch 2, accuracy 0.959600
# epoch 3, accuracy 0.966200
# epoch 4, accuracy 0.972100
# epoch 5, accuracy 0.977000
# epoch 6, accuracy 0.977200
# epoch 7, accuracy 0.977400
# epoch 8, accuracy 0.979000
# epoch 9, accuracy 0.980600
# epoch 10, accuracy 0.979700
# epoch 11, accuracy 0.980000
# epoch 12, accuracy 0.981400
# epoch 13, accuracy 0.981800
# epoch 14, accuracy 0.981400
# epoch 15, accuracy 0.981600
# epoch 16, accuracy 0.980500
# epoch 17, accuracy 0.982300
# epoch 18, accuracy 0.982400
# epoch 19, accuracy 0.982400
# epoch 20, accuracy 0.983100
# epoch 21, accuracy 0.982300
# epoch 22, accuracy 0.982100
# epoch 23, accuracy 0.983700
# epoch 24, accuracy 0.984200
# epoch 25, accuracy 0.983200
# epoch 26, accuracy 0.983000
# epoch 27, accuracy 0.983400
# epoch 28, accuracy 0.983500
# epoch 29, accuracy 0.984100
# epoch 30, accuracy 0.983600
# epoch 31, accuracy 0.983900
# epoch 32, accuracy 0.983900
# epoch 33, accuracy 0.984200
# epoch 34, accuracy 0.983300
# epoch 35, accuracy 0.983000
# epoch 36, accuracy 0.983900
# epoch 37, accuracy 0.983800
# epoch 38, accuracy 0.984400
# epoch 39, accuracy 0.983300
# epoch 40, accuracy 0.984500
# epoch 41, accuracy 0.984700
# epoch 42, accuracy 0.984200
# epoch 43, accuracy 0.984100
# epoch 44, accuracy 0.983600
# epoch 45, accuracy 0.984500
# epoch 46, accuracy 0.984300
# epoch 47, accuracy 0.984700
# epoch 48, accuracy 0.984400
# epoch 49, accuracy 0.984000
# epoch 50, accuracy 0.984700

# FashionMNIST (1 hidden layer with 1024 nodes):
# epoch 1, accuracy 0.714800
# epoch 2, accuracy 0.772000
# epoch 3, accuracy 0.819900
# epoch 4, accuracy 0.850700
# epoch 5, accuracy 0.824000
# epoch 6, accuracy 0.819900
# epoch 7, accuracy 0.835400
# epoch 8, accuracy 0.830700
# epoch 9, accuracy 0.869100
# epoch 10, accuracy 0.867100
# epoch 11, accuracy 0.866300
# epoch 12, accuracy 0.869200
# epoch 13, accuracy 0.841100
# epoch 14, accuracy 0.882400
# epoch 15, accuracy 0.881600
# epoch 16, accuracy 0.878000
# epoch 17, accuracy 0.881400
# epoch 18, accuracy 0.863800
# epoch 19, accuracy 0.852200
# epoch 20, accuracy 0.874900
# epoch 21, accuracy 0.880200
# epoch 22, accuracy 0.883600
# epoch 23, accuracy 0.856700
# epoch 24, accuracy 0.874600
# epoch 25, accuracy 0.881700
# epoch 26, accuracy 0.885600
# epoch 27, accuracy 0.858000
# epoch 28, accuracy 0.859700
# epoch 29, accuracy 0.876800
# epoch 30, accuracy 0.865400
# epoch 31, accuracy 0.873600
# epoch 32, accuracy 0.887400
# epoch 33, accuracy 0.880800
# epoch 34, accuracy 0.882900
# epoch 35, accuracy 0.853500
# epoch 36, accuracy 0.875700
# epoch 37, accuracy 0.882000
# epoch 38, accuracy 0.867900
# epoch 39, accuracy 0.861900
# epoch 40, accuracy 0.886800
# epoch 41, accuracy 0.879800
# epoch 42, accuracy 0.879300
# epoch 43, accuracy 0.886600
# epoch 44, accuracy 0.896100
# epoch 45, accuracy 0.891600
# epoch 46, accuracy 0.896000
# epoch 47, accuracy 0.885100
# epoch 48, accuracy 0.890000
# epoch 49, accuracy 0.887800
# epoch 50, accuracy 0.891800

# FashionMNIST (2 hidden layer with 512/64 nodes):
# epoch 1, accuracy 0.741900
# epoch 2, accuracy 0.816500
# epoch 3, accuracy 0.767900
# epoch 4, accuracy 0.824900
# epoch 5, accuracy 0.848100
# epoch 6, accuracy 0.829100
# epoch 7, accuracy 0.831600
# epoch 8, accuracy 0.855900
# epoch 9, accuracy 0.856300
# epoch 10, accuracy 0.849800
# epoch 11, accuracy 0.835000
# epoch 12, accuracy 0.820900
# epoch 13, accuracy 0.859500
# epoch 14, accuracy 0.864500
# epoch 15, accuracy 0.816900
# epoch 16, accuracy 0.878400
# epoch 17, accuracy 0.876900
# epoch 18, accuracy 0.857400
# epoch 19, accuracy 0.868400
# epoch 20, accuracy 0.878900
# epoch 21, accuracy 0.874100
# epoch 22, accuracy 0.879500
# epoch 23, accuracy 0.886400
# epoch 24, accuracy 0.882800
# epoch 25, accuracy 0.802400
# epoch 26, accuracy 0.868700
# epoch 27, accuracy 0.856600
# epoch 28, accuracy 0.870600
# epoch 29, accuracy 0.847200
# epoch 30, accuracy 0.891900
# epoch 31, accuracy 0.867200
# epoch 32, accuracy 0.886700
# epoch 33, accuracy 0.872300
# epoch 34, accuracy 0.889800
# epoch 35, accuracy 0.865700
# epoch 36, accuracy 0.878600
# epoch 37, accuracy 0.886300
# epoch 38, accuracy 0.817000
# epoch 39, accuracy 0.890900
# epoch 40, accuracy 0.887800
# epoch 41, accuracy 0.888500
# epoch 42, accuracy 0.888500
# epoch 43, accuracy 0.881900
# epoch 44, accuracy 0.871400
# epoch 45, accuracy 0.882000
# epoch 46, accuracy 0.882500
# epoch 47, accuracy 0.891500
# epoch 48, accuracy 0.861400
# epoch 49, accuracy 0.890700
# epoch 50, accuracy 0.879900

# FashionMNIST (2 hidden layer with 512/64 nodes and dropout):
# epoch 1, accuracy 0.718700
# epoch 2, accuracy 0.707500
# epoch 3, accuracy 0.794500
# epoch 4, accuracy 0.775100
# epoch 5, accuracy 0.826400
# epoch 6, accuracy 0.835900
# epoch 7, accuracy 0.816900
# epoch 8, accuracy 0.855300
# epoch 9, accuracy 0.854000
# epoch 10, accuracy 0.865100
# epoch 11, accuracy 0.857000
# epoch 12, accuracy 0.862900
# epoch 13, accuracy 0.858900
# epoch 14, accuracy 0.856200
# epoch 15, accuracy 0.875300
# epoch 16, accuracy 0.873500
# epoch 17, accuracy 0.848000
# epoch 18, accuracy 0.866000
# epoch 19, accuracy 0.865500
# epoch 20, accuracy 0.869800
# epoch 21, accuracy 0.862300
# epoch 22, accuracy 0.853400
# epoch 23, accuracy 0.884100
# epoch 24, accuracy 0.870700
# epoch 25, accuracy 0.877100
# epoch 26, accuracy 0.878400
# epoch 27, accuracy 0.880400
# epoch 28, accuracy 0.881500
# epoch 29, accuracy 0.877600
# epoch 30, accuracy 0.865700
# epoch 31, accuracy 0.855600
# epoch 32, accuracy 0.871700
# epoch 33, accuracy 0.881400
# epoch 34, accuracy 0.866400
# epoch 35, accuracy 0.879900
# epoch 36, accuracy 0.875600
# epoch 37, accuracy 0.862400
# epoch 38, accuracy 0.855000
# epoch 39, accuracy 0.887700
# epoch 40, accuracy 0.887200
# epoch 41, accuracy 0.873100
# epoch 42, accuracy 0.880400
# epoch 43, accuracy 0.854500
# epoch 44, accuracy 0.886200
# epoch 45, accuracy 0.870700
# epoch 46, accuracy 0.876000
# epoch 47, accuracy 0.889900
# epoch 48, accuracy 0.886100
# epoch 49, accuracy 0.880600
# epoch 50, accuracy 0.888900
