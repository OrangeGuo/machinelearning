from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

x = np.r_[[[0, 1], [1, 1], [1, 0]]]
y = [0, 1, 2]
# clf = svm.SVC(kernel='linear')
# clf.fit(x, y)
#
# print clf.support_vectors_
# print clf.support_
# print clf.n_support_
# print clf.predict([[0, 2]])
# plt.plot([0,1],[1,0])
# plt.show()
print x[:,0]
print x[:,1]
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()