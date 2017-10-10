import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(0)

def hidden_activate(x):
    try:
        result = np.zeros((len(x), len(x[0])))
    except TypeError:
        result = np.zeros((len(x)))
        for i in range(len(x)):
            result[i] = np.tanh(x[i])
        return result
    for i in(range(len(x))):
            for j in(range(len(x[0]))):
                result[i][j] = math.tanh(x[i][j])
    return result

def output_activate(x):
    try:
        result = np.zeros((len(x),len(x[0])))
    except TypeError:
        result = np.zeros((len(x)))
        for i in range(len(x)):
            result[i] = 1.0/(1.0+np.exp(-x[i]))
        return result
    for i in(range(len(x))):
        for j in(range(len(x[0]))):
            result[i][j] = 1.0/(1.0+np.exp(-x[i][j]))
    return result

def matrix_init(row, col):
    matrix = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            matrix[i][j] = np.random.uniform(-1,1)
    return matrix

x =  matrix_init(3,2)
# print x
# print hidden_activate(x)
# print output_activate(x)
# xx = np.linspace(-5,5)
# plt.plot(xx,hidden_activate(xx))
# plt.plot(xx,output_activate(xx))
# plt.show()

