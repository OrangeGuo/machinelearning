from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]] # funciton like reshape
# x =[ np.random.randn(20,2)-[2,2]]*1+[np.random.randn(20,2)+[2,2]]*1
y = [0]*20 + [1]*20
clf = svm.SVC(kernel='linear')
clf.fit(x,y)
w = clf.coef_[0] #faxiangliang
a = -w[0]/w[1]
# xx = np.linspace(-5,5,2)
xx = np.r_[-5. ,5.]
yy = a*xx-(clf.intercept_[0])/w[1]
b = clf.support_vectors_[0]
yy_down = a*xx + (b[1]-a*b[0])
b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1]-a*b[0])

print "W:",w
print "a:",a
print "support_vectors_:",clf.support_vectors_
print "clf.coef_:",clf.coef_
plt.figure(figsize=(8,8))
plt.plot(xx,yy)
plt.plot(xx,yy_down)
plt.plot(xx,yy_up)
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Paired)
plt.axis('tight')
plt.show()
