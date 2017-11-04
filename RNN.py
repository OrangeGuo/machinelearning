# coding:utf-8
import string
import sys
import numpy as np
import sklearn.datasets
import math

epsilion = 0.01
reg_lambda = 0.01
np.random.seed(0)

input_layers = 1
hidden_layers = 5
output_layers = 1
X = []
y = []
loop = 0
temp = 0.0
index = 0.0
# path = 'data/' + str(sys.argv[1])
for line in open('/home/orange/Workspaces/MyEclipse 2015/SoftwareReliabilityTest/data/CSR3.DAT'):
    s = line.strip().split()
    if s:
        loop += 1
        temp += string.atof(s[0])
        index += string.atof(s[1])
        X.append(np.array([temp]))
        y.append(index)
X = np.array(X)
y = np.array(y)
sample_num = loop
X_max = X[loop - 1][0]
X_min = X[0][0]
y_max = y[loop - 1]
y_min = y[0]


# y =np.array([y]).T
def net_init():
    # sample, result = sklearn.datasets.make_moons(sample_num, noise=0.20)
    sample = (X - X_min) / (X_max - X_min)
    result = (y - y_min) / (y_max - y_min)
    W1 = 2 * np.random.random((hidden_layers, input_layers)) - 1
    W2 = 2 * np.random.random((output_layers, hidden_layers)) - 1
    H = 2*np.random.random((hidden_layers,hidden_layers)) - 1
    S = np.zeros((hidden_layers,input_layers))
    b1 = np.zeros((hidden_layers, 1))
    b2 = np.zeros((output_layers, 1))
    net = {'sample': sample, 'result': result, 'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2,'H':H,'S':S}
    return net


def train(net):
    sample, result, W1, W2, b1, b2, S, H= net['sample'], net['result'], net['W1'], net['W2'], net['b1'], net['b2'],net['S'],net['H']
    for i in range(loop):
        z2 = W1*X[i][0]+ b1+H.dot(S)
        S =  z2
        a2 = np.tanh(z2)
        z3 = W2.dot(a2) + b2
        exp_scores = np.exp(z3)

        delta3 = exp_scores - result[i]
        # delta3[result,range(sample_num)]-=1
        # print delta3[result,range(sample_num)]-2
        dW2 = delta3.dot(a2.T)
        db2 = np.sum(delta3, axis=1, keepdims=True)
        delta2 = W2.T.dot(delta3) * (1 - np.power(a2, 2))
        dW1 = delta2*X[i][0]
        dH = delta2.dot(S.T)
        db1 = np.sum(delta2, axis=1, keepdims=True)

        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
        dH += reg_lambda * H

        W1 += -epsilion * dW1
        W2 += -epsilion * dW2
        H += -epsilion * dH
        b1 += -epsilion * db1
        b2 += -epsilion * db2
        net = {'sample': sample, 'result': result, 'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2,'H':H,'S':S}

    return net


def predict(net):
    sample, result, W1, W2, b1, b2,S,H = net['sample'], net['result'], net['W1'], net['W2'], net['b1'], net['b2'],net['S'],net['H']
    for i in range(loop):
        z2 = W1.dot(sample[i][0]) + b1,
        a2 = np.tanh(z2)
        z3 = W2.dot(a2) + b2
        exp_scores = np.exp(z3)
        exp_scores = exp_scores * (y_max - y_min) + y_min
        print exp_scores

    # print exp_scores
    # with open('data/BPnetwork.txt', 'w') as file:
    #     # s = '\t\r'.join(str(i) for i in (y))
    #     # file.write(s)
    #     # file.write('\r')
    #     s = '\t\r'.join(str(i) for i in (np.array(exp_scores[0])))
    #     file.write(s)
    #     file.close()
        # print exp_scores

net = net_init()
output = train(net)
predict(output)