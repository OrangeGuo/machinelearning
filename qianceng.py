import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import sklearn.datasets
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)

plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
 # plt.show()

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X, y)


def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


plot_decision_boundary(lambda x: clf.predict(x))
plt.title('logistic regression')
# plt.show()
num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 2
epsilion = 0.01  # learning rate
reg_lambda = 0.01


def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z2 = W1.dot(X.T) + b1
    a2 = np.tanh(z2)
    z3 = W2.dot(a2) + b2
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    corect_logprobs = -np.log(probs[y, range(num_examples)])
    data_loss = np.sum(corect_logprobs)
    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1.0 / num_examples * data_loss


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z2 = W1.dot(x.T) + b1
    a2 = np.tanh(z2)
    z3 = W2.dot(a2) + b2
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    return np.argmax(probs, axis=0)


def build_model(nn_hdim, num_passes=20000, print_loss=False):
    np.random.seed(0)
    W1 = np.random.randn(nn_hdim, nn_input_dim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((nn_hdim, 1))
    W2 = np.random.randn(nn_output_dim, nn_hdim) / np.sqrt(nn_hdim)
    b2 = np.zeros((nn_output_dim, 1))
    model = {}

    for i in xrange(0, num_passes):

        # Forward propagation
        z2 = W1.dot(X.T) + b1
        a2 = np.tanh(z2)
        z3 = W2.dot(a2) + b2
        exp_scores = np.exp(z3)
        probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

        # Back propagation
        delta3 = probs
        delta3[y, range(num_examples)] -= 1
        dW2 = delta3.dot(a2.T)
        db2 = np.sum(delta3, axis=1, keepdims=True)
        delta2 = W2.T.dot(delta3) * (1 - np.power(a2, 2))
        dW1 = delta2.dot(X)
        db1 = np.sum(delta2, axis=1, keepdims=True)

        # add regularization term
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # gradient descent parameter update
        W1 += -epsilion * dW1
        b1 += -epsilion * db1
        W2 += -epsilion * dW2
        b2 += -epsilion * db2

        # assign new parameter to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        if print_loss and i % 1000 == 0:
            print("loss after iteration %i: %f" % (i, calculate_loss(model)))

    return model


model = build_model(3, print_loss=True)
plot_decision_boundary(lambda x: predict(model, x))
plt.title('decision boundary for hidden layer size 3')
plt.show()
