from sklearn import datasets
from sklearn import cross_validation
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
# 导入训练数据集
diabetes = datasets.load_diabetes()
diabetes_feature = diabetes.data[:, np.newaxis, 2]
diabetes_target = diabetes.target
train_feature, test_feature, train_target, test_target = cross_validation.train_test_split(diabetes_feature,
                                                                                           diabetes_target,
                                                                                           test_size=0.33,
                                                                                           random_state=56)


# 创建模型
model = linear_model.LinearRegression()
model.fit(train_feature, train_target)

# 绘制图像
plt.scatter(train_feature, train_target, color='black')
plt.scatter(test_feature, train_target, color='red')
plt.plot(test_feature, model.predict(test_feature), color='blue', linewidth=3)
plt.legend(('Fit line', 'Train set', 'Test set'), loc='lower right')
plt.title('LinearRegressionExample')
plt.xticks(())
plt.yticks(())
plt.show()
