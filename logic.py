# import numpy as np
# import math
#
# np.random.seed(0)
#
# def getMatrix(row, col):
#     return np.random.random((row, col))
#
# def sigmoid(x):
#     return math.tanh(x)
#
# def dsigmoid(y):
#     return 1.0 - y**2
#
# class Net:
#     def __init__(self, input_layers, hidden_layers, _output_layers):
#         self.input_layers = input_layers + 1
#         self.hidden_layers = hidden_layers
#         self.output_layers = _output_layers
#
#         self.ainput_layers = [1.0] * self.input_layers
#         self.ahidden_layers = [1.0] * self.hidden_layers
#         self.aoutput_layers = [1.0] * self.output_layers
#
#         self.weights_input = getMatrix(self.input_layers, self.hidden_layers)
#         self.weights_output = getMatrix(self.hidden_layers, self.output_layers)
#
#         self.c_input = getMatrix(self.input_layers, self.hidden_layers)
#         self.c_output = getMatrix(self.hidden_layers, self.output_layers)
#
#     def update(self, inputs):
#         if len(inputs) != self.input_layers - 1:
#             raise ValueError('input error!')
#         for i in range(self.input_layers-1):
#              self.ainput_layers[i] = inputs[i]
#         self.ahidden_layers = sigmoid(np.dot(self.ainput_layers, self.weights_input))
#         self.aoutput_layers = sigmoid(np.dot(self.ahidden_layers, self.weights_output))
#
#         return  self.aoutput_layers
#
#     def backPropagate(self, targets, ):
import math
import random
import string

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m

#  def makeMatrix(row, col):
#     return random.random((row, col))

def sigmoid(x):
    return math.tanh(x)


def dsigmoid(y):
    return 1.0 - y ** 2


class NN:

    def __init__(self, ni, nh, no):
        self.ni = ni + 1
        self.nh = nh
        self.no = no

        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('!!!!')

        for i in range(self.ni - 1):
            # self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('!!!')

        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change
                # print(N*change, M*self.co[j][k])

        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)


def demo():
    pat = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    target =[[0,1,1,0]].T
    n = NN(2, 2, 1)
    n.train(pat)
    n.test(target)
    # n.weights()


if __name__ == '__main__':
    demo()


