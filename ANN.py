import numpy as np
from matplotlib import pyplot as plt

class layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilites = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        self.output  = probabilites
class loss:
    def forward(self,inputs,target):
        sum = 0
        for i in range(0,len(target)):
            sum = sum + np.log(inputs[i]) * target[0][i]
        self.output = -sum
class ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

#Trying
X = np.array([[1,0],
              [2,0]])
target = np.zeros((10,1)).T
target[0][0]=1

layer1 = layer(2, 10)
layer2 = layer(10,10)
activation1 = ReLU()
activation2 = softmax()
loss1 = loss()


layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

loss1.forward(activation2.output[0],target)
print(loss1.output)

