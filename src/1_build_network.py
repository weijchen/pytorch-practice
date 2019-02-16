
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
