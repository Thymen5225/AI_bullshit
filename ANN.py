import numpy as np
from matplotlib import pyplot as plt

class layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
layer1 = layer(2,2)
X = np.array([[1,2],
              [3,4]])
layer1.forward(X)
print(layer1.output)