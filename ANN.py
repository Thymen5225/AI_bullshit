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
        res=[]
        for j in inputs:
            sum = 0
            for i in range(0,len(j)):
                sum = sum + np.log(j[i]) * target[0][i]
            res.append(-sum)
        self.output = res
class ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class onehot:
    def forward(self,samples):
        one_hot = np.zeros((samples.size,samples.max()+1))
        one_hot[np.arange(samples.size),samples] = 1
        one_hot = one_hot.T
        self.output = one_hot
def der_ReLU(Z):
    return Z>0

class back_prop:
    def forward(self,Z1,A1,Z2,A2,W2,X,Y):
        m = Y.size
        one_hot = onehot()
        one_hot.forward(Y)
        dZ2 = A2 - one_hot.output
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * der_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        self.output =  dW1,db1,dW2,db2





#Trying
X = np.array([[1,0],
              [2,0]])
target = np.zeros((10,1)).T
target[0][0]=1

layer1 = layer(2, 2)
layer2 = layer(2,2)
activation1 = ReLU()
activation2 = softmax()
loss1 = loss()
learning_rate = 0.1


layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

one_hot = onehot()
one_hot.forward(np.arange(0,2))

update=back_prop()
update.forward(layer1.output,activation1.output,layer2.output,activation2.output,layer2.weights,X,np.arange(0,2))
print(update.output)
# A = activation2.output
# print(A)
# print(one_hot.output)