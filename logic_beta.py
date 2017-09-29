import numpy as np
import math
import  random
np.random.seed(0)

def getMatrix(row, col):
    return np.random.random((row, col))

def sigmoid(x):
    return math.tanh(x)

def dsigmoid(y):
    return 1.0 - y**2

class Net:
    def __init__(self, input_layers, hidden_layers, _output_layers):
        self.input_layers = input_layers + 1
        self.hidden_layers = hidden_layers
        self.output_layers = _output_layers

        self.ainput_layers = np.zeros((4,self.input_layers))

        self.weights_input = getMatrix(self.input_layers, self.hidden_layers)
        self.weights_output = getMatrix(self.hidden_layers, self.output_layers)
        for i in range(self.input_layers):
            for j in range(self.hidden_layers):
                self.weights_input[i][j] = np.random.uniform(-0.2,0.2)
        for i in range(self.hidden_layers):
            for j in range(self.output_layers):
                self.weights_output[i][j] = np.random.uniform(-0.2,0.2)
        self.c_input = getMatrix(self.input_layers, self.hidden_layers)
        self.c_output = getMatrix(self.hidden_layers, self.output_layers)

    def update(self, inputs):
        if len(inputs[0]) != self.input_layers - 1:
            raise ValueError('input error!')

        for i in range(len(inputs)):
            for j in range(self.input_layers-1):
                self.ainput_layers[i][j] = inputs[i][j]
        self.ahidden_layers = np.dot(self.ainput_layers,self.weights_input)
        # self.ahidden_layers = np.sum(self.ahidden_layers, axis=0, keepdims=True)
        for i in range(len(self.ahidden_layers)):
            for j in range(len(self.ahidden_layers[0])):
                self.ahidden_layers[i][j]  = sigmoid(self.ahidden_layers[i][j])
        # self.aoutput_layers = np.dot( list(map(list,zip(*self.ahidden_layers))),self.weights_output)
        self.aoutput_layers = np.dot(self.ahidden_layers,self.weights_output)
        # self.aoutput_layers = np.sum(self.aoutput_layers, axis=0, keepdims=True)
        # print self.aoutput_layers
        for i in range(len(self.aoutput_layers)):
            self.aoutput_layers[i][0] = sigmoid(self.aoutput_layers[i][0])
        # print self.aoutput_layers
        return  self.aoutput_layers

    def backPropagate(self, targets, output, N, M):
        output_delta = np.zeros((4,1))

        for i in range(len(targets[0])):
            output_delta[i][0] = dsigmoid(output[i][0])*(targets[0][i] - output[i][0])
        # print targets
        # print output
        # print output_delta
        for s in range(4):
            hidden_delta = np.zeros((1, self.hidden_layers))
            for i in range(self.hidden_layers):
                error = 0.0
                for j in range(self.output_layers):
                    error += output_delta[s][j] * self.weights_output[i][j]
                hidden_delta[0][i] = dsigmoid(self.ahidden_layers[s][i]) * error
            # print hidden_delta
            # print output_delta
            # print self.weights_output
            # print self.weights_input
            # break
            for i in range(self.hidden_layers):
                for j in range(self.output_layers):
                    change = output_delta[s][j] * self.ahidden_layers[s][i]
                    self.weights_output[i][j] = self.weights_output[i][j] + N * change + M * self.c_output[i][j]
                    self.c_output[i][j] = change
            for i in range(self.input_layers - 1):
                for j in range(self.hidden_layers):
                    change = hidden_delta[0][j] * self.ainput_layers[s][i]
                    self.weights_input[i][j] = self.weights_input[i][j] + N * change + M * self.c_input[i][j]
                    self.c_input[i][j] = change
            # self.weights()
        error = 0.0
        for k in range(len(targets[0])):
            error += 0.5*(targets[0][k] - self.aoutput_layers[k][0])**2
        return error
    def weights(self):
        print 'weight_input:'
        print self.weights_input
        print 'weight_output'
        print self.weights_output
    def train(self, samples, results, iteration=1000, N=0.5, M=0.1):
        for i in range(iteration):
            error = 0.0
            output = self.update(inputs=samples)
            error += self.backPropagate(results, output,N,M)
            # break
            if i%100 == 0:
                print('error %-.5f' % error)
    def  predict(self,input):
        print self.update(inputs=input)

sample = [[0,0],
                   [0,1],
                   [1,0],
                   [1,1]]
result = [[0,1,1,0]]
net = Net(2,2,1)
# print type(sample)
net.train(sample, result)
net.predict(sample)