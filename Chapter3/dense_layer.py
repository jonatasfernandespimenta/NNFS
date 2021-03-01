import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=10, classes=3)

class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.001 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
    pass

  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases
    pass

dense1 = Layer_Dense(2, 3)
dense1.forward(X)

print(dense1.output[:5])
