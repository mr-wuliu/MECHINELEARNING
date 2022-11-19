import pandas as pd
import numpy as np

"""
Clementine Model
模块用于数据处理. 也部分作为公共库, 例如损失函数, 梯度等
Clementine 意为小橘子, 谐音局子, 表示写这个玩意儿像坐牢一样.
至于为什么不叫tangerine, 因为已经有这个库的名字了, 防止重名.
"""

'''
公共函数, 包含:
sigmoid() 激活函数
cost() 计算损失
nabla() 用解析的方式计算梯度
scrubbing() 数据加常数项, 并可选归一化
wolfe() 搜索合适的alpha值
predict() 结果预测
shuffle() 数据打乱
'''


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def cost(X, theta, Y, c=False, Lambda=0):
    """
    损失函数, 采用负对数似然函数作为衡量标准
    正则化项采用L2正则化, 默认正则化项为零
    :param X: 样本集
    :param theta: 待定系数
    :param Y: 标签集
    :param c: 函数系数
    :param Lambda: 拟合程度
    :return:
    """
    #h = sigmoid(X @ theta)
    if c:
        # loss = (-1.0 / m) * np.sum(Y.T @ np.log(h + 1e-6)
        #                            + (1 - Y).T @ np.log(1 - h + 1e-6)) \
        #        + Lambda * theta.T @ theta / 2
        m, n = X.shape
        a = np.multiply(X @ theta, Y)
        b = np.log(1 + np.exp(X @ theta))
        loss = (1.0 / m) * np.sum(-a + b) \
               + Lambda * theta.T @ theta / 2
    else:
        # loss = -1.0 * np.sum(Y.T @ np.log(h + 1e-8)
        #                      + (1 - Y).T @ np.log(1 - h + 1e-8)) \
        #        + Lambda * theta.T @ theta / 2
        a = np.multiply(X @  theta, Y)
        b = np.log(1 + np.exp(X @ theta))
        loss = np.sum( -a  + b )\
                +Lambda * theta.T @ theta / 2
    return loss


def nabla(X, theta, Y, Lambda=0, c=False):
    """
    实现nabla算子
    :param X:
    :param theta:
    :param Y:
    :param Lambda: 正则化程度
    :param c: 损失函数系数
    :return:
    """
    grad = X.T @ (sigmoid(X @ theta) - Y) + Lambda * theta
    if c:
        m, n = X.shape
        return (1.0 / m) * grad
    return grad


def scrubbing(data, stander=True):
    """
    对数据做归一化处理并且在第一列添加常数项
    :param data:
    :param stander:
    :return:
    """
    row, columns = data.shape
    ones = np.ones((row, 1))
    if stander:
        x = data
        data = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    data = np.append(ones, data, axis=1)
    return data


def wolfe(x, theta, y, d, c, Lambda, log=False):
    c1 = 0.1
    c2 = 0.9
    alpha = 25
    a = 0
    b = 50
    max_epoch = 40
    epoch = 0
    while a < b and epoch < max_epoch:
        if cost(x, theta + alpha * d, y, c=c, Lambda=Lambda) <= \
                cost(x, theta, y, c=c, Lambda=Lambda) + \
                c1 * alpha * nabla(x, theta, y, c=c, Lambda=Lambda).T @ d:
            if nabla(x, theta + alpha * d, y, c=c, Lambda=Lambda).T @ d >= \
                    c2 * nabla(x, theta, y, c=c, Lambda=Lambda).T @ d:
                break
            else:
                a = alpha
        else:
            b = alpha
        alpha = (a + b) / 2
        epoch += 1
    if log and epoch == max_epoch:
        alpha = 0.05
        print("wolfe搜索失败")
    return alpha

def predict(X, theta, y):
    m, n = X.shape
    pre = np.zeros((m, 1))

    for i in range(m):
        activate = sigmoid(np.dot(X[i, :], theta))
        if activate > 0.5:
            pre[i] = 1
    count = 0
    for i in range(m):
        if pre[i] == y[i]:
            count += 1
    return count / m


def shuffle(*args):
    """
    输入给定的数据集, 全部以同一方式打乱
    :return: 返回打乱
    """
    data= []  # 不定长数据集
    m, n = args[0].shape
    shuffle_ix = np.random.permutation(np.arange(m))
    for i in args:
        data.append(args[i][shuffle_ix])
    return data


'''
数据读取
'''

def load_bp(percent=0.8, stander=False,is_shuffle=True):
    x_row = np.array(pd.read_csv('./data/X_data.csv',header=None))
    y_row = pd.read_csv('./data/y_label.csv',header=None)
    x_row = scrubbing(x_row, stander=stander)
    # y_row进行独热编码
    y_row = np.array(pd.get_dummies(
        y_row,
        columns=[0]
    ))
    if is_shuffle:
        data = shuffle(x_row,y_row)
        x = data[0]
        y = data[1]
        row = x.shape[0]
        x_train = x[:int(percent * row), :]
        y_train = y[:int(percent * row), :]
        x_test = x[int(percent * row):, :]
        y_test = y[int(percent * row):, :]
        return x_train, y_train, x_test, y_test
    else :
        return x_row, y_row



def load_telco(percent=0.8, stander=True):
    pd_reader = pd.read_csv("./data/telco.csv")
    # 用平均值对空值进行填充
    pd_reader = pd_reader.fillna(pd_reader.mean())
    result = np.array(pd_reader, dtype=np.float64)
    rows, columns = result.shape
    beta = percent  # 训练集占比
    # 训练集
    X_tr = scrubbing(result[:int(rows * beta), :columns - 1], stander)
    y_tr = result[:int(rows * beta), -1:]

    # 测试集
    X_test = scrubbing(result[int(rows * beta):, :columns - 1], stander)
    y_test = result[int(rows * beta):, -1:]
    return X_tr, y_tr, X_test, y_test


def load_lr(percent=0.8, stander=True):
    lr = np.array(pd.read_table('./data/LR_data.txt'))
    rows, columns = lr.shape
    beta = percent  # 训练集占比
    # 训练集
    X_tr = scrubbing(lr[:int(rows * beta), :columns - 1], stander)
    y_tr = lr[:int(rows * beta), -1:]

    # 测试集
    X_test = scrubbing(lr[int(rows * beta):, :columns - 1], stander)
    y_test = lr[int(rows * beta):, -1:]
    return X_tr, y_tr, X_test, y_test


def load_cancer_amazon(filename="cancer", stander=True):
    if filename == "cancer":
        pd_reader_x = pd.read_csv("./data/cancer_X.csv", header=None)
        pd_reader_y = pd.read_csv("./data/cancer_y.csv", header=None)
        if stander:
            x = scrubbing(np.array(pd_reader_x)[:, :172])
            y = np.array(pd_reader_y)
        else:
            x = np.array(pd_reader_x)[:, :172]
            y = np.array(pd_reader_y)
    elif filename == "amazon":
        pd_reader_x = pd.read_csv("./data/amazon_X.csv", header=None)
        pd_reader_y = pd.read_csv("./data/amazon_y.csv", header=None)
        if stander:
            x = scrubbing(np.array(pd_reader_x))
            y = np.array(pd_reader_y)
        else:
            x = np.array(pd_reader_x)
            y = np.array(pd_reader_y)
    else:
        x = None
        y = None
        print("没有该文件")
    return x, y


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_bp(is_shuffle=True)
    print(y_test)