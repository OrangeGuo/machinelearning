import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron

df = pd.read_csv("data.csv", header=0)
# print df
feature = df[["x", "y"]]
target = df["class"]
train_feature, test_feature, train_target, test_target = train_test_split(feature, target, train_size=0.77)
model = Perceptron()
model.fit(train_feature, train_target)
model2 = LinearSVC()
model2.fit(train_feature, train_target)
print model.score(test_feature, test_target)
print model2.score(test_feature, test_target)
