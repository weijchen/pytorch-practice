# -*- coding: utf-8 -*-
"""
    * Author: Jimmy Chen
    * PN: PyTorch tutorial - Training on CPU
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

# ===== download db =====
transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./db', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./db', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# # ----- show image -----
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # get some random training images
# dataiter = iter(trainloader)
# inputs, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(inputs))

# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# ----- build the network -----
print("> Building the network...")

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

model = Net()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# ----- train the network -----
print("> Training the network...")
total_step = len(trainloader)
for epoch in range(2):
  running_loss = 0.0
  for i, (inputs, labels) in enumerate(trainloader):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    if i % 2000 == 1999:
      print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
           .format(epoch+1, 2, i+1, total_step, loss.item()))
print('> Finish Training')

# ----- testing the network -----
print("> Testing the network...")
model.eval()

with torch.no_grad():
  correct = 0
  total = 0
  for inputs, labels in testloader:
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print('> Finish Testing')

with torch.no_grad():
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
