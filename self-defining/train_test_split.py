import numpy as np
# X：原始数据集
# y：原始数据集的 label
# test_ratio：测试数据集占原始数据集的比例
# seed：随机种子
def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据X和y按照test_ratio分割成X_train, X_test, y_train, y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size if must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ratio must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)   # 测试数据集大小
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]  # 训练数据集
    y_train = y[train_indexes]

    X_test = X[test_indexes]    # 测试数据集
    y_test= y[test_indexes]

    return X_train, X_test, y_train, y_test