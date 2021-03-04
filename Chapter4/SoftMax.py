import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=10, classes=3)

class Activation_ReLU:
  def forward(self, inputs):
    self.output = np.maximum(0, inputs)

class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.001 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
    pass

  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases
    pass

class Activation_Softmax:
  def forward(self, inputs):
    exp_values = np.sum(input, np.max(input, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    self.outputs = probabilities

dense1 = Layer_Dense(2, 3)

activation1 = Activation_ReLU()
dense1.forward(X)

activation1.forward(dense1.output)

dense2 = Layer_Dense(3, 3)

activation2 = Activation_Softmax()
dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.outputs[:5])
