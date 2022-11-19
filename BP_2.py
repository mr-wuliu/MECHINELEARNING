import pandas as pd
import numpy as np
import datetime


def sigmoid(x):
    return 1.0 /(1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 1.初始化参数
def initialize_parameters(n_x, n_h, n_y):
    # np.random.seed(1)
    w1 = np.random.randn(n_h, n_x) * 0.01
    w2 = np.random.randn(n_y, n_h) * 0.01
    parameters = {'w1': w1, 'w2': w2}
    return parameters


# 2.前向传播
def forward_propagation(X, parameters):
    w1 = parameters['w1']
    w2 = parameters['w2']
    z1 = np.dot(w1, X)
    # a1 = np.tanh(z1)
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache


# 3.计算代价函数
def compute_cost(a2, Y, parameters):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(a2+1e-6), Y) + np.multiply((1 - Y), np.log(1 - a2 + 1e-6))
    cost = - np.sum(logprobs) / m
    return cost


def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]

    w2 = parameters['w2']

    a1 = cache['a1']
    a2 = cache['a2']

    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    dz1 = np.multiply(np.dot(w2.T, dz2),d_sigmoid(a1))
    dw1 = (1 / m) * np.dot(dz1, X.T)

    grads = {'dw1': dw1, 'dw2': dw2}

    return grads


def update_parameters(parameters, grads, learning_rate=0.02):
    w1 = parameters['w1']
    w2 = parameters['w2']

    dw1 = grads['dw1']
    dw2 = grads['dw2']
    # 更新参数
    w1 = w1 - dw1 * learning_rate
    w2 = w2 - dw2 * learning_rate

    parameters = {'w1': w1, 'w2': w2}

    return parameters


def nn_model(X, Y, n_h, n_input, n_output,
             num_iterations=10000, print_cost=False, learning_rate=0.02):
    # np.random.seed(3)
    n_x = n_input
    n_y = n_output
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 梯度下降循环
    for i in range(0, num_iterations):
        if i > 4400:
            learning_rate=0.0075
        a2, cache = forward_propagation(X, parameters)
        cost = compute_cost(a2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads,learning_rate=learning_rate)
        if print_cost and i % 1000 == 0:
            print('迭代第%i次，代价函数为：%f' % (i, cost))
    return parameters


def predict(parameters, x_test, y_test):
    w1 = parameters['w1']
    w2 = parameters['w2']

    z1 = np.dot(w1, x_test)
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)

    n_rows = y_test.shape[0]
    n_cols = y_test.shape[1]
    # 预测值结果存储
    output = np.empty(shape=(n_rows, n_cols), dtype=int)

    for i in range(n_cols):
        # 将每条测试数据的预测结果（概率）存为一个行向量
        temp = np.zeros(shape=n_rows)
        for j in range(n_rows):
            temp[j] = a2[j][i]

        # 将每条结果（概率）从小到大排序，并获得相应下标
        sorted_dist = np.argsort(temp)
        length = len(sorted_dist)

        # 将概率最大的置为1，其它置为0
        for k in range(length):
            if k == sorted_dist[length - 1]:
                output[k][i] = 1
            else:
                output[k][i] = 0

    count = 0
    for k in range(0, n_cols):
        if output[0][k] == y_test[0][k] and output[1][k] == y_test[1][k] and output[2][k] == y_test[2][k]:
            count = count + 1

    acc = count / int(y_test.shape[1]) * 100
    print('预测准确率：%.2f%%' % acc)


if __name__ == "__main__":
    X_row = np.array(pd.read_csv('./data/X_data.csv',header=None))
    y_row = pd.DataFrame(pd.read_csv('./data/y_label.csv', header=None))

    ones = np.ones((X_row.shape[0], 1))
    X = np.append(ones, X_row,axis=1)

    Y = np.array(pd.get_dummies(
        y_row,
        columns=[0]
    ))
    # 建立打乱序列, shuffle_ix 可以令x, y以相同的顺序打乱
    shuffle_ix = np.random.permutation(np.arange(len(X)))
    x = X[shuffle_ix]
    y = Y[shuffle_ix]

    row, columns = x.shape
    percent = 1
    # 划分训练集和测试集

    x_train = x[:int(percent * row), :].T
    y_train = y[:int(percent * row), :].T

    x_test = x[int(percent * row):, :].T
    y_test = y[int(percent * row):, :].T

    # 开始训练
    start_time = datetime.datetime.now()
    # 输入400个节点，隐层25个节点，输出10个节点
    network = nn_model(x_train, y_train,
                       n_h=26, n_input=401, n_output=10,
                       num_iterations=50000, print_cost=True,learning_rate=0.02)
    end_time = datetime.datetime.now()
    print("用时：" + str((end_time - start_time).seconds)+'s')
    predict(network, x_train, y_train)
