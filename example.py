from sklearn import linear_model

# 第一步：调用一个机器方法构建相应模型model，并设置模型参数
model = linear_model.LinearRegression()
# 第二步：使用该机器模型提供的model.fit()，训练模型
model.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
# 第三步：使用该机器模型提供的model.predic()，用于预测

print(model.predict([4, 4]))
