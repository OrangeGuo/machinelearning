#coding:utf-8
import  string
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
epsilion = 0.01
reg_lambda = 0.01
np.random.seed(0)

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
        # y.append(string.atof(s[1]))
        # temp.append(string.atof(s[1]))

        X.append(np.array([temp]))
        y.append(index)
X = np.array(X)
y = np.array(y)
# print y
def net_init():
    # sample, result = sklearn.datasets.make_moons(sample_num, noise=0.20)
    sample =X
    result = y
    W1 = np.random.randn(hidden_layers_num, input_layers_num) / np.sqrt(input_layers_num)
    b1 = np.zeros((hidden_layers_num, 1))
    W2 = np.random.randn(output_layers_num, hidden_layers_num) / np.sqrt(hidden_layers_num)
    b2 = np.zeros((output_layers_num, 1))
    net = {'sample':sample,'result':result,'W1':W1,'W2':W2,'b1':b1,'b2':b2}
    return net

def train(net,loops=2):
    sample,result,W1,W2,b1,b2 = net['sample'],net['result'],net['W1'],net['W2'],net['b1'],net['b2']
    for i in range(loops):
        # z2 = W1.dot(sample.T)+b1
        # a2 = np.tanh(z2)
        l2 = 1/(1+np.exp(-(W1.dot(sample.T))))
        z3  = W2.dot(l2)
        print z3
        # exp_scores = np.exp(z3)
        # probs = exp_scores/
        #
        # delta3 = probs
        # delta3[result,range(sample_num)]-=1
        # print delta3[result,range(sample_num)]-2
        delta3 = z3 -result
        dW2 = delta3.dot(l2.T)
        # db2 = np.sum(delta3,axis=1,keepdims=True)
        delta2 = W2.T.dot(delta3)*(1-np.power(l2,2))
        dW1 = delta2.dot(sample)
        # db1 = np.sum(delta2,axis=1,keepdims=True)

        dW2 += reg_lambda*W2
        dW1 += reg_lambda*W1

        W1 += -epsilion*dW1
        W2 += -epsilion*dW2
        # b1 += -epsilion*db1
        # b2 += -epsilion*db2
        # net = {'sample': sample, 'result': result, 'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
        net = {'sample': sample, 'result': result, 'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

        # print("loss after iteration %i: %f" % (i,data_loss ))
        # axis.append(data_loss)
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
    sample = np.array([np.array([1])])
    z2 = W1.dot(sample.T)+b1
    a2 = np.tanh(z2)
    z3 = W2.dot(a2)+b2
    print z3
    print result[0]
    # exp_scores = np.exp(z3)
    # probs =exp_scores/ np.sum(exp_scores,axis=0,keepdims=True)
    # s = np.argmax(probs,axis=0)
    # for j in range(sample_num):
    #     if(s[j] != result[j]):
    #         s[j] = -1
    # plt.figure('Predict')
    # plt.scatter(sample[:,0],sample[:,1],c=s)
    # plt.show()
    # return np.argmax(probs, axis=0)

# net = net_init()
# output = train(net)
# predict(net)

# print X
# sample, result = sklearn.datasets.make_moons(sample_num, noise=0.20)
# print result
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
print X