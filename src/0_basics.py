import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# # ===== tensor =====
# # Construct an empty matrix
# x = torch.empty(5, 3)
# print(x)

# # Construct a randomly initialized matrix
# y = torch.rand(5, 3)
# print(y)

# # Construct a tensor directly from data
# z = torch.tensor([5.5, 3])
# print(z)

# x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
# print(x)

# x = torch.randn_like(x, dtype=torch.float)    # override dtype!
# print(x)                                      # result has the same size

# print(x.size())                               # get size

# # Resize (.view())
# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# print(x.size(), y.size(), z.size())

# # Get the value (.item())
# x = torch.randn(1)
# print(x)
# print(x.item())

# # ===== NumPy Bridge =====
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)

# a = np.ones(5)
# b = torch.from_numpy(a)
# np.add(a, 1, out=a)
# print(a)
# print(b)

# # ===== CUDA Tensors =====
# # let us run this cell only if CUDA is available
# # We will use ``torch.device`` objects to move tensors in and out of GPU
# if torch.cuda.is_available():
#   device = torch.device("cuda")          # a CUDA device object
#   y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
#   x = x.to(device)                       # or just use strings ``.to("cuda")``
#   z = x + y
#   print(z)
#   print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

# # ===== AutoGrad =====
# x = torch.ones(2, 2, requires_grad=True)
# print(x)
# y = x + 2
# print(y)

# # Function and Tensors encode a complete history of computation
# print(y.grad_fn)

# z = y * y * 3
# out = z.mean()

# print(z, out)

# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)    # change an existing Tensor's requires_grad flag in-place
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)

# # ===== Gradients =====
# out.backward()
# print(x.grad)

# ===== Neural Networks =====
# ----- Constructing NN -----
# class Net(nn.Module):

#   def __init__(self):
#     super(Net, self).__init__()
#     # 1 input image channel, 6 output channels, 5*5 square convolution
#     # Kernel 1
#     self.conv1 = nn.Conv2d(1, 6, 5)
#     self.conv2 = nn.Conv2d(6, 16, 5)
#     # an affine operation: y = Wx + b
#     self.fc1 = nn.Linear(16 * 5 * 5, 120)
#     self.fc2 = nn.Linear(120, 84)
#     self.fc3 = nn.Linear(84, 10)

#   def forward(self, x):
#     # Max pooling over a (2, 2) window
#     x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#     # If the size is a square you can only specify a single number
#     x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#     x = x.view(-1, self.num_flat_features(x))
#     x = F.relu(self.fc1(x))
#     x = F.relu(self.fc2(x))
#     x = self.fc3(x)
#     return x

#   def num_flat_features(self, x):
#     size = x.size()[1:]
#     num_features = 1
#     for s in size:
#       num_features *= s
#     return num_features

# net = Net()
# print(net)

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

# ===== Training a classifier =====
# ----- download db -----
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

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
inputs, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(inputs))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# ----- build the network -----
class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    # 3 input image channel, 6 output channels, 5*5 square convolution
    # Kernel 1
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    # Max pooling over a (2, 2) window
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# print("> Building the network...")
# net = Net()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# print("> Training the network...")
# # ----- train the network -----
# for epoch in range(2):
#   running_loss = 0.0
#   for i, data in enumerate(trainloader, 0):
#     # get the inputs
#     inputs, labels = data

#     # zero the parameter gradients
#     optimizer.zero_grad()

#     # forward + backward + optimize
#     outputs = net(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()

#     # print statistics
#     running_loss += loss.item()
#     if i % 2000 == 1999:  # print every 2000 mini-batches
#       print('[%d %5d] loss: %.3f' %
#             (epoch + 1, i + 1, running_loss / 2000))
#       running_loss = 0.0

# print('Finish Training')

# # ----- testing the network -----
# print("> Testing the network...")
# dataiter = iter(testloader)
# inputs, labels = dataiter.next()

# # print images
# # imshow(torchvision.utils.make_grid(inputs))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# outputs = net(inputs)

# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         inputs, labels = data
#         outputs = net(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))

# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         inputs, labels = data
#         outputs = net(inputs)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1

# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))

# ===== Training on GPU =====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print("> Building the network...")
net = Net()
net.to(device)

# model = Model(input_size, output_size)
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   model = nn.DataParallel(model)

# model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("> Training the network...")
# ----- train the network -----
for epoch in range(2):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    # get the inputs
    inputs, labels = data
    # inputs = inputs.to(device)
    inputs, labels = inputs.to(device), labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 2000 == 1999:  # print every 2000 mini-batches
      print('[%d %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0

print('Finish Training')

# ----- testing the network -----
print("> Testing the network...")
dataiter = iter(testloader)
inputs, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(inputs)

_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # inputs = inputs.cuda()

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # inputs = inputs.cuda()

        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# # ===== Basic autograd example 1 =====
# # Create tensors
# x = torch.tensor(1., requires_grad=True)
# w = torch.tensor(2., requires_grad=True)
# b = torch.tensor(3., requires_grad=True)

# # Build a computational graph
# y = w * x + b  # y = 2 * x + 3

# # Compute gradients
# y.backward()

# # Print out the gradients
# print(x.grad)
# print(w.grad)
# print(b.grad)