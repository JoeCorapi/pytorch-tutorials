# torch.autograd is PyTorch’s automatic differentiation engine that powers neural network training.
# Neural Networks are a collection of nested functions that execute on an input - defined by
# parameters (weights and biases) and stored in Tensors
# Two Steps:
# Foreward Propagation: NN runs the data through functions and makes guesses about the correct output
# Backward Propagation: Adjust parameters proportionate to error. Builds derivates from error with respect to gradients
# - Optimizes parameters using gradient descent for next iteration

import torch
from torchvision.models import resnet18, ResNet18_Weights

# Load up pre trained model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
# 3 channels, height/width of 64
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# Run foreward pass to make a prediction
prediction = model(data)
# Run backward pass, stores gradients for each model param in .grad
loss = (prediction - labels).sum()
loss.backward()

# Load an optimizer, with learning rate 0.01 and momentum 0.9 (look this up)
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# Call step to initiate gradient descent, adjusting each param stored in error tensor by it's .grad props
optim.step()

# Differentiation
