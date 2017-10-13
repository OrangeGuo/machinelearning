import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

input_layers_num = 2
hidden_layers_num = 4
output_layers_num = 1
sample_num = 4
# X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# y = np.array([[0, 1, 1, 0]]).T
np.random.seed(0)



sample, result = sklearn.datasets.make_moons(sample_num, noise=0.20)
syn0 = 2 * np.random.random((input_layers_num, hidden_layers_num)) - 1
syn1 = 2 * np.random.random((hidden_layers_num, output_layers_num)) - 1
result =  np.array([np.array(result)]).T

def train(syn0,syn1,loops=1000):
    for i in range(loops):
        l1 = 1 / (1 + np.exp(-(np.dot(sample, syn0))))
        l2 = 1 / (1 + np.exp(-(np.dot(l1, syn1))))
        l2_delta = (result - l2) * (l2 * (1 - l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1 - l1))
        syn1 += l1.T.dot(l2_delta)
        syn0 += sample.T.dot(l1_delta)
    return  l2

output = train(syn0,syn1).reshape(sample_num)
plt.figure('predict')
plt.bar(range(len(output)),output)
plt.title('outputs')
plt.show()
