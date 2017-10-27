#coding:utf-8
import  string

# X = [1, 2]
# state = [0.0, 0.0]
# # 分开定义不同输入部分的权重方便操作
# w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
# w_cell_input = np.asarray([0.5, 0.6])
# b_cell = np.asarray([0.1, -0.1])
#
# # 定义用于输出的全连接层参数
# w_output = np.asarray([[1.0], [2.0]])
# b_output = 0.1
#
# # 按照时间顺序执行循环神经网络的前向传播过程
# for i in range(len(X)):
#     # 计算循环体中的全连接层神经网络
#     before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
#     state = np.tanh(before_activation)
#     # 根据当前时刻状态计算最终输出
#     final_output = np.dot(state, w_output) + b_output
#     print("output: h", i, final_output)
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import  math
input_layers_num = 2
hidden_layers_num = 8
output_layers_num = 1
sample_num = 1
epsilion = 0.01
reg_lambda = 0.01
# X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# y = np.array([[0, 1, 1, 0]]).T
np.random.seed(0)

X = [[]]
y = []
loop = 0
for line in open('/home/orange/Workspaces/MyEclipse 2015/SoftwareReliabilityTest/data/CSR3.DAT'):
    s = line.strip().split()
    temp = []
    if s :
        loop+=1
        temp.append(string.atof(s[0]))
        y.append(string.atof(s[1]))
        temp.append(string.atof(s[1]))
        X.append(temp)
def net_init():
    sample, result = sklearn.datasets.make_moons(sample_num, noise=0.20)
    W1 = 2 * np.random.random(( hidden_layers_num,input_layers_num)) - 1
    W2 = 2 * np.random.random((output_layers_num,hidden_layers_num)) - 1
    b1 = np.zeros((hidden_layers_num,1))
    b2 = np.zeros((output_layers_num,1))
    S = np.zeros((hidden_layers_num,sample_num))
    net = {'sample':sample,'result':result,'W1':W1,'W2':W2,'b1':b1,'b2':b2,"S":S}
    return net

def train(net,loops=loop):
    W1,W2,b1,b2,S= net['W1'],net['W2'],net['b1'],net['b2'],net["S"]
    axis = []
    for i in range(loops):
        sample = np.zeros((sample_num,input_layers_num))
        temp = np.array(X[i+1])
        sample[0][0] = temp[0]
        sample[0][1]= temp[1]
        result = y[i]
        z2 = W1.dot(sample.T)+b1+S
        a2 = np.tanh(z2)
        S = z2
        z3  = W2.dot(a2)+b2
        # exp_scores = np.exp(z3)
        # print  exp_scores
        # probs = exp_scores/np.sum(exp_scores,axis=0,keepdims=True)
        # print  probs
        delta3 = z3
        data_loss = 0.0
        for j in range(sample_num):
            delta3[0][j] -= result
            data_loss += delta3[0][j]
        # delta3[result,range(sample_num)]-=1
        # print delta3[result,range(sample_num)]-1
        dW2 = delta3.dot(a2.T)
        db2 = np.sum(delta3,axis=1,keepdims=True)
        delta2 = W2.T.dot(delta3)*(1-np.power(a2,2))
        dW1 = delta2.dot(sample)
        db1 = np.sum(delta2,axis=1,keepdims=True)

        dW2 += reg_lambda*W2
        dW1 += reg_lambda*W1

        W1 += -epsilion*dW1
        W2 += -epsilion*dW2
        b1 += -epsilion*db1
        b2 += -epsilion*db2
        net = {'sample': sample, 'result': result, 'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

        print("loss after iteration %i: %f" % (i,data_loss ))
        axis.append(data_loss)
    # plt.figure('Gradient Desent')
    # plt.bar(range(len(axis)),axis)
    # with open('/home/orange/Workspaces/MyEclipse 2015/SoftwareReliabilityTest/b.txt','w') as file:
    #     s= '\t\r'.join(str(i) for i in axis)
    #     file.write(s)
    #     file.close()
    return net

def loss_function(net):
    sample,result,W1,W2,b1,b2 =net['sample'],net['result'], net['W1'],net['W2'],net['b1'],net['b2']
    z2 = W1.dot(sample.T)+b1
    a2 = np.tanh(z2)
    z3 = W2.dot(a2)+b2
    # exp_scores = np.exp(z3)
    probs = z3/np.sum(z3,axis=0,keepdims=True)
    # corect_logprobs = math.fabs(probs)
    # data_loss = np.sum(corect_logprobs)
    # data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    # print data_loss
    # return 1.0 / sample_num * data_loss

def predict(net):
    sample,result,W1,W2,b1,b2 =net['sample'],net['result'],net['W1'],net['W2'],net['b1'],net['b2']
    z2 = W1.dot(sample.T)+b1
    a2 = np.tanh(z2)
    z3 = W2.dot(a2)+b2
    exp_scores = np.exp(z3)
    probs =exp_scores/ np.sum(exp_scores,axis=0,keepdims=True)
    s = np.argmax(probs,axis=0)
    for j in range(sample_num):
        if(s[j] != result[j]):
            s[j] = -1
    plt.figure('Predict')
    plt.scatter(sample[:,0],sample[:,1],c=s)
    plt.show()
    return np.argmax(probs, axis=0)

net = net_init()
output = train(net)
predict(net)

# sample, result = sklearn.datasets.make_moons(sample_num, noise=0.20)
# print result
