import numpy as np
from matplotlib import pyplot as plt
import image_processing_fleet
from PIL import Image, ImageOps

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
        dZ1 = W2.dot(dZ2.T) * der_ReLU(Z1).T
        dW1 = 1 / m * (dZ1).dot(X)
        dW1 = dW1.T
        db1 = 1 / m * np.sum(dZ1)
        self.output =  dW1,db1,dW2,db2

class update:
    def forward(self,W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2.T
        b2 = b2 - alpha * db2
        self.output = W1, b1, W2, b2

resolution = [64,36] #64,36
X1, fleet, Y1 = image_processing_fleet.getImageData(resolution[0],resolution[1])
X1 = np.array(X1)
Y1 = np.array(Y1)
Y2 = []
for i in Y1:
    empty = np.zeros((1,10))
    empty[0][i] = 1
    Y2.append(empty)
Y2 = np.array(Y2).reshape(len(Y2),10)
Y2 = Y2.astype(int)

X_test, fleet, Y1_test = image_processing_fleet.getTestData(resolution[0],resolution[1])
X_test = np.array(X_test)
Y1_test = np.array(Y1_test)
Y2_test = []
for i in Y1_test:
    empty = np.zeros((1,10))
    empty[0][i] = 1
    Y2_test.append(empty)
Y2_test = np.array(Y2_test).reshape(len(Y2_test),10)
Y2_test= Y2_test.astype(int)


X = X1
Y = Y2
learning_rate = 0.0001 #0.0001
iterations = 2500 #2500
Neurons = 200 #200

layer1 = layer(resolution[0]*resolution[1], Neurons) #resolution[0]*resolution[1]
layer2 = layer(Neurons,10)
activation1 = ReLU()
activation2 = softmax()

def run():
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
        if i/iterations*100 % 5 ==0:
            print((i/iterations)*100)

    results=[]
    for i in activation2.output:
        #print(np.argmax(i))
        results.append(np.argmax(i))
    positive = 0
    mistakes = 0
    for i in range(0,len(Y1)):
        a = Y1[i] - results[i]
        if a==0:
            positive = positive +1
        else:
            mistakes = mistakes +1
    print("Learning data:",positive,mistakes,positive/len(Y1)*100)

    layer1.forward(X_test)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    results_test=[]
    for i in activation2.output:
        results_test.append(np.argmax(i))
        positive_t = 0
        mistakes_t = 0
    for i in range(0, len(Y1_test)):
        b = Y1_test[i] - results_test[i]
        if b == 0:
            positive_t = positive_t + 1
        else:
            mistakes_t = mistakes_t + 1
    print("Test data:", positive_t, mistakes_t, positive_t / len(Y2_test) * 100)

run()
#print("Y1:",Y1.shape,"Y1_test:",Y1_test.shape,"Y2:",Y2.shape,"Y2_test:",Y2_test.shape)