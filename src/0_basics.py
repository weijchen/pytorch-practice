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

# ----- Resize (.view()) -----
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# ----- Get the value (.item()) -----
x = torch.randn(1)
print(x)
print(x.item())

# ===== NumPy Bridge =====
# ----- torch -> numpy -----
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# ----- numpy -> torch -----
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# ===== CUDA Tensors =====
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
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

# ===== Basic autograd example 2 =====
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

# # ===== Gradients =====
# out.backward()
# print(x.grad)

# params = list(net.parameters())
# print(len(params))
# print(params[0].size())                      # conv1's weight

# input = torch.randn(1, 1, 32, 32)
# out = net(input)
# print(out)

# net.zero_grad()  # zeroes the gradient buffers of all parameters
# out.backward(torch.randn(1, 10))

# # ----- Loss Function -----
# output = net(input)
# target = torch.randn(10)     # a dummy target, for example
# target = target.view(1, -1)  # make it the same shape as output
# criterion = nn.MSELoss()

# loss = criterion(output, target)
# print(loss)
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# # ----- Backprop -----
# net.zero_grad()  # zeroes the gradient buffers of all parameters

# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)

# loss.backward()

# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)

# # ===== Update the weights =====
# # ----- Hand craft (SGD) -----
# learning_rate = 0.01
# for f in net.parameters():
#   f.data.sub_(f.grad.data * learning_rate)

# # ----- Call by built-in function -----
# # create your optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.01)

# # in your training loop
# optimizer.zero_grad()
# output = net(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()     # dose the update
