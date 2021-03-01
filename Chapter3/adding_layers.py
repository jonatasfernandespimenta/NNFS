import numpy as np

inputs = [
	[1, 4, 5, 3],
	[0.2, -0.5, 1.4, 1],
	[-0.5, 5, 2, 0.3]
]

weights = [
	[0.3, -0.1, 0.6, 2],
	[-0.5, 0.9, 5, 4],
	[-0.9, 1, 0.4, 3]
]
bias = [1, -4.0, 0.5]

weights2 = [
	[0.7, 0.5, -0.6],
	[-2.5, 0.2, 2],
	[4.9, -1.3, 0.9]
]
bias2 = [8, -4.2, 0.3]


layer1_outputs = np.dot(inputs, np.array(weights).T) + bias
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + bias2


print(layer2_outputs)
