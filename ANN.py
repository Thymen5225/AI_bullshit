import numpy as np
from matplotlib import pyplot as plt

class layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn((n_inputs,n_neurons))
        self.biases = np.zeros((1,n_neurons))