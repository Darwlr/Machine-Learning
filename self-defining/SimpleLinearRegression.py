import numpy as np
from metrics import r2_score
# 使用向量化运算
class SimpleLinearRegression:
    def __init__(self):
        """初始化 Simple Linear Regression 模型"""
        self.a_ = None
        self.b_ = None
    def fit(self, x_train, y_train):
        """根据训练数据集 x_train, y_train训练模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean) #分子点乘
        d = (x_train - x_mean).dot(x_train - x_mean) #分母点乘

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self
    def predict(self, x_predict): # x_predict 为一个向量
        """给定预测数据集x_predict, 返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
        return np.array([self._predict(x) for x in x_predict])
    def _predict(self, x_single): # x_single 为一个数
        """给定单个预测数据x_single, 返回x_single的预测结果值"""
        return self.a_ * x_single + self.b_
    def score(self, x_test, y_test):
        """根据测试数据集x_test和y_test确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)
    def __repr__(self):
        return "SimpleLinearRegression()"

# import numpy as np
#
# class SimpleLinearRegression1:
#     def __init__(self):
#         """初始化 Simple Linear Regression 模型"""
#         self.a_ = None
#         self.b_ = None
#     def fit(self, x_train, y_train):
#         """根据训练数据集 x_train, y_train训练模型"""
#         assert x_train.ndim == 1, \
#             "Simple Linear Regression can only solve single feature training data"
#         assert len(x_train) == len(y_train), \
#             "the size of x_train must be equal to the size of y_train"
#
#         x_mean = np.mean(x_train)
#         y_mean = np.mean(y_train)
#
#         num = 0.0
#         d = 0.0
#         for x, y in zip(x_train, y_train):
#             num += (x - x_mean) * (y - y_mean)
#             d += (x - x_mean) ** 2
#         self.a_ = num / d
#         self.b_ = y_mean - self.a_ * x_mean
#
#         return self
#     def predict(self, x_predict): # x_predict 为一个向量
#         """给定预测数据集x_predict, 返回表示x_predict的结果向量"""
#         assert x_predict.ndim == 1, \
#             "Simple Linear Regression can only solve single feature training data"
#         assert self.a_ is not None and self.b_ is not None, \
#             "must fit before predict!"
#         return np.array([self._predict(x) for x in x_predict])
#     def _predict(self, x_single): # x_single 为一个数
#         """给定单个预测数据x_single, 返回x_single的预测结果值"""
#         return self.a_ * x_single + self.b_
#     def __repr__(self):
#         return "SimpleLinearRegression1()"
#
# # 使用向量化运算
# # 只需要改变 fit 函数
# class SimpleLinearRegression2:
#     def __init__(self):
#         """初始化 Simple Linear Regression 模型"""
#         self.a_ = None
#         self.b_ = None
#     def fit(self, x_train, y_train):
#         """根据训练数据集 x_train, y_train训练模型"""
#         assert x_train.ndim == 1, \
#             "Simple Linear Regression can only solve single feature training data"
#         assert len(x_train) == len(y_train), \
#             "the size of x_train must be equal to the size of y_train"
#
#         x_mean = np.mean(x_train)
#         y_mean = np.mean(y_train)
#
#         num = (x_train - x_mean).dot(y_train - y_mean) #分子点乘
#         d = (x_train - x_mean).dot(x_train - x_mean) #分母点乘
#
#         self.a_ = num / d
#         self.b_ = y_mean - self.a_ * x_mean
#
#         return self
#     def predict(self, x_predict): # x_predict 为一个向量
#         """给定预测数据集x_predict, 返回表示x_predict的结果向量"""
#         assert x_predict.ndim == 1, \
#             "Simple Linear Regression can only solve single feature training data"
#         assert self.a_ is not None and self.b_ is not None, \
#             "must fit before predict!"
#         return np.array([self._predict(x) for x in x_predict])
#     def _predict(self, x_single): # x_single 为一个数
#         """给定单个预测数据x_single, 返回x_single的预测结果值"""
#         return self.a_ * x_single + self.b_
#     def __repr__(self):
#         return "SimpleLinearRegression2()"
