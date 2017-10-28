
from numpy import *
import numpy as np
import string
input_layers_num = 1
hidden_layers_num = 5
output_layers_num = 1
X = []
y = []
loop = 0
temp = 0.0
index = 0.0
for line in open('/home/orange/Workspaces/MyEclipse 2015/SoftwareReliabilityTest/data/CSR3.DAT'):
    s = line.strip().split()
    if s :
        loop+=1
        temp += string.atof(s[0])
        index += string.atof(s[1])
        X.append(np.array([temp]))
        y.append(index)
X = np.array(X)
y = np.array(y)
y =np.array([y]).T
input_weights = 2*np.random.random((input_layers_num,hidden_layers_num)) - 1
hidden = X.dot(input_weights)
output_weights = (mat(hidden)).I.dot(y)
sample =np.array([[100]])
print sample.dot(input_weights).dot(output_weights)


