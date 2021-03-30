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
X1, carrier_lst, Y1 = image_processing_fleet.getImageData(resolution[0],resolution[1])
X1 = np.array(X1)
Y1 = np.array(Y1)
Y2 = []
for i in Y1:
    empty = np.zeros((1,10))
    empty[0][i] = 1
    Y2.append(empty)
Y2 = np.array(Y2).reshape(len(Y2),10)
Y2 = Y2.astype(int)

test = Image.open('testfile.png')
test_gray = ImageOps.grayscale(test.resize((resolution[0],resolution[1]), Image.ANTIALIAS))
X_test = np.asarray(test_gray).flatten()


#Trying
# X = np.array([[1,0,2,5,6],
#               [2,0,3,6,7],
#               [2,1,2,6,2],
#               [3,1,5,1,0]])
# X = X
# Y = np.zeros((10,4)).T
# Y[0][0]=1
# Y[1][2]=1
# Y[2][2]=1
# Y[3][3]=1
# Y = Y.astype(int)
# print(X)
# print(X1)
X = X1
Y = Y2


layer1 = layer(resolution[0]*resolution[1], 70) #resolution[0]*resolution[1]
layer2 = layer(70,10)
activation1 = ReLU()
activation2 = softmax()
learning_rate = 0.0001
iterations = 2000

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
    for i in activation2.output:
        print("Result:",np.argmax(i))

run()
