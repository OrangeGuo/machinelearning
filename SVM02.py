from sklearn import datasets
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

digits = datasets.load_digits()
# for index, image in enumerate(digits.images[:5]):
#     plt.subplot(2, 5, index + 1)
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
# plt.show()
feature = digits.data
target = digits.target
train_feature, test_feature, train_target, test_target = train_test_split(feature, target, test_size=0.33)
model = SVC(gamma=0.001)
model.fit(train_feature, train_target)
results = model.predict(test_feature)
scores = accuracy_score(test_target, results)
print scores

