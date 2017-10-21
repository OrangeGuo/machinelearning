import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

input_layers_num = 2
hidden_layers_num = 8
output_layers_num = 2
sample_num = 200
epsilion = 0.01
reg_lambda = 0.01
# X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# y = np.array([[0, 1, 1, 0]]).T
np.random.seed(0)



def net_init():
    sample, result = sklearn.datasets.make_moons(sample_num, noise=0.20)
    W1 = 2 * np.random.random(( hidden_layers_num,input_layers_num)) - 1
    W2 = 2 * np.random.random((output_layers_num,hidden_layers_num)) - 1
    b1 = np.zeros((hidden_layers_num,1))
    b2 = np.zeros((output_layers_num,1))
    net = {'sample':sample,'result':result,'W1':W1,'W2':W2,'b1':b1,'b2':b2}
    return net

def train(net,loops=20000):
    sample,result,W1,W2,b1,b2 = net['sample'],net['result'],net['W1'],net['W2'],net['b1'],net['b2']
    axis = []
    for i in range(loops):
        z2 = W1.dot(sample.T)+b1
        a2 = np.tanh(z2)
        z3  = W2.dot(a2)+b2
        exp_scores = np.exp(z3)
        probs = exp_scores/np.sum(exp_scores,axis=0,keepdims=True)

        delta3 = probs
        delta3[result,range(sample_num)]-=1
        # print delta3[result,range(sample_num)]-2
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
        if i % 1000 == 0:
            print("loss after iteration %i: %f" % (i, loss_function(net)))
            axis.append(loss_function(net))
    plt.figure('Gradient Desent')
    plt.bar(range(len(axis)),axis)
    with open('/home/orange/Workspaces/MyEclipse 2015/SoftwareReliabilityTest/b.txt','w') as file:
        s= '\t\r'.join(str(i) for i in axis)
        file.write(s)
        file.close()
    return net

def loss_function(net):
    sample,result,W1,W2,b1,b2 =net['sample'],net['result'], net['W1'],net['W2'],net['b1'],net['b2']
    z2 = W1.dot(sample.T)+b1
    a2 = np.tanh(z2)
    z3 = W2.dot(a2)+b2
    exp_scores = np.exp(z3)
    probs = exp_scores/np.sum(exp_scores,axis=0,keepdims=True)
    corect_logprobs = -np.log(probs[result, range(sample_num)])
    data_loss = np.sum(corect_logprobs)
    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1.0 / sample_num * data_loss

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
# plt.figure('predict')
# plt.bar(range(len(output)),output)
# plt.title('outputs')
# plt.show()