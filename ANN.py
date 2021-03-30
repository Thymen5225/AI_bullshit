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
class ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

def der_ReLU(Z):
    return Z>0

class back_prop:
    def forward(self,Z1,A1,Z2,A2,W2,X,Y):
        m =Y.size
        dZ2 = A2 - Y
        dW2 = 1 / m * (dZ2.T).dot(A1)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2.T) * der_ReLU(Z1).T
        dW1 = 1 / m * (dZ1).dot(X)
        dW1 = dW1.T
        db1 = 1 / m * np.sum(dZ1)
        self.output =  dW1,db1,dW2,db2

class update:
    def forward(self,W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        self.output = W1, b1, W2, b2


#Trying
X = np.array([[1,0,2],
              [2,0,3],
              [2,2,2]])
Y = np.zeros((10,3)).T
Y[0][0]=1
Y[1][2]=1
Y[2][9]=1
Y = Y.astype(int)

layer1 = layer(3, 10)
layer2 = layer(10,10)
activation1 = ReLU()
activation2 = softmax()
learning_rate = 0.01
iterations = 100000
for i in range(0,iterations):
    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    change = back_prop()
    change.forward(layer1.output,activation1.output,layer2.output,activation2.output,layer2.weights,X,Y)
    dW1, db1, dW2, db2 = change.output
    correction = update()
    correction.forward(layer1.weights,layer1.biases,layer2.weights,layer2.biases,dW1,db1,dW2,db2,learning_rate)
    W1, b1, W2, b2 = correction.output
    layer1.weights = W1
    layer2.weights = W2
    layer1.biases = b1
    layer2.biases = b2

print(activation2.output)
for i in activation2.output:
    print(np.argmax(i))

print(Y)
