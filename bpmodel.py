import numpy as np
import matplotlib.pyplot as plt

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0, 0, 1, 1]]).T
np.random.seed(1)
syn0 = 2*np.random.random((3,1)) - 1
l0= X
l2 = nonlin(np.dot(l0, syn0))
l1_error = y - l2
l1_delta = l1_error * nonlin(l2, True)
syn0 += np.dot(l0.T, l1_delta)
for iter in xrange(10000):
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l1_error = y - l1
    l1_delta = l1_error * nonlin(l1, True)
    syn0 += np.dot(l0.T, l1_delta)

print "resutl:"
print  l1
print l1.reshape(4)
# print np.random.random(2)
# plt.bar(range(len(y)),l2)
# plt.show()
total_width, n, x = 0.8, 3 ,4
width = total_width / n
x = x - (total_width - width) / 2

plt.bar( x,l2.T.reshape(4,order='C'),  width=width, label='a')
# plt.bar(x + width, l1, width=width, label='b')
# plt.bar(x + 2 * width, y, width=width, label='c')
plt.legend()
plt.show()