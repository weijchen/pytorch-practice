# -*- coding: utf-8 -*-
"""
    * Author: Jimmy Chen
    * PN: PyTorch tutorial - Basic operations
    * Ref:
      - https://github.com/yunjey/pytorch-tutorial
      - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    * Log: 
      - Created Feb. 2019
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# ===== Tensor =====
# Construct an empty matrix
x = torch.empty(5, 3)
print(x)

# Construct a randomly initialized matrix
y = torch.rand(5, 3)
print(y)

# Construct a tensor directly from data
z = torch.tensor([5.5, 3])
print(z)

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

print(x.size())                               # get size

# Resize (.view())
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# Get the value (.item())
x = torch.randn(1)
print(x)
print(x.item())

# ===== Tensor & NumPy Transformation =====
# torch -> numpy
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# numpy -> torch
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# ===== CUDA Tensors =====
if torch.cuda.is_available():
  device = torch.device("cuda")          # a CUDA device object
  y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
  x = x.to(device)                       # or just use strings ``.to("cuda")``
  z = x + y
  print(z)
  print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

# ===== AutoGrad example 1 =====
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)

# Function and Tensors encode a complete history of computation
print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)    # change an existing Tensor's requires_grad flag in-place
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# ===== AutoGrad example 2 =====
# Create tensors
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph
y = w * x + b  # y = 2 * x + 3

# Compute gradients
y.backward()

# Print out the gradients
print(x.grad)
print(w.grad)
print(b.grad)

# ===== Basic autograd example 3 =====
# Create tensors
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Build a fully connected layer
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

# Build loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass
pred = linear(x)

# Compute loss
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward pass
loss.backward()

# Print out the gradients
print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# 1-step gradient descent
optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())

# ===== Input pipeline =====

transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./db', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64,
                                          shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./db', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Fetch one data pair (read data from disk).
image, label = train_dataset[0]
print(image.size())
print(label)

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = data_iter.next()

# Actual usage of the data loader is as below.
for images, labels in train_loader:
    # Training code should be written here.
    pass

# ===== Input pipeline for custom dataset =====
# You should build your custom dataset as below.
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self):
    # TODO
    # 1. Initialize file paths or a list of file names. 
    pass
  def __getitem__(self, index):
    # TODO
    # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
    # 2. Preprocess the data (e.g. torchvision.Transform).
    # 3. Return a data pair (e.g. image and label).
    pass
  def __len__(self):
    # You should change 0 to the total size of your dataset.
    return 0 

custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=64, shuffle=True)

# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())     # (64, 100)

# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))








