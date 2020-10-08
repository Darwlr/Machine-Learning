# import numpy as np
# from math import sqrt
# from collections import Counter
#
# def kNN_classify(k, X_train, y_train, x):
#     assert 1 <= k <= X_train.shape[0], "k must be valid"
#     assert X_train.shape[0] == y_train.shape[0], \
#         "the size of X_train must equal to the size of y_train"
#     assert X_train.shape[1] == x.shape[0], \
#         "the feature number of x must be equal to X_train"
#
#     distance = [ sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]
#     nearest = np.argsort(distance)
#
#     topK_y = [ y_train[i] for i in nearest[:k] ]
#     votes = Counter(topK_y)
#
#     return votes.most_common(1)[0][0]

# 重新整理kNN算法
import numpy as np
from collections import Counter
from math import sqrt
from metrics import accuracy_score

class kNNClassifier:
    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_tarin = None
        self._y_train = None
    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        self._X_train = X_train
        self._y_train = y_train
        return self
    def predict(self, X_predict):
        """给定待预测数据集X_predict,返回表示X_predict的结果向量"""
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)
    def _predict(self, x):
        """给定单个待预测数据x，返回x_predict的预测结果值"""
        distances = [ sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train ]
        nearest = np.argsort(distances)
        topK_y = [ self._y_train[i] for i in nearest[:self.k] ]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]
    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
    def __repr__(self):
        return "KNN(k=%d)" % self.k