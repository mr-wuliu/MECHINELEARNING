import numpy as np
import pandas as pd
import time


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def init_network(n_x, n_h, n_y):
    theta_1 = np.random.randn(n_x, n_h) * 0.01
    theta_2 = np.random.randn(n_h, n_y) * 0.01
    network = {'theta_1': theta_1, 'theta_2': theta_2}
    return network


def forward_propagation(X, network):
    theta_1 = network['theta_1']
    theta_2 = network['theta_2']
    z1 = X @ theta_1
    a1 = sigmoid(z1)
    z2 = a1 @ theta_2
    a2 = sigmoid(z2)

    trace = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return trace


def loss_cross_entropy(y_pre, Y):
    # 交叉熵
    m = Y.shape[1]
    logprobs = np.multiply(np.log(y_pre + 1e-6), Y) + np.multiply((1 - Y), np.log(1 - y_pre + 1e-6))
    cost = - np.sum(logprobs) / m
    return cost


def backward_propagation(network, trace, X, Y):
    m = Y.shape[1]

    theta_2 = network['theta_2']

    a1 = trace['a1']
    a2 = trace['a2']

    d_z_2 = a2 - Y
    d_theta_2 = (1 / m) * a1.T @ d_z_2
    d_z_1 = np.multiply(d_z_2 @ theta_2.T,
                                  d_sigmoid(a1))
    d_theta_1 = (1 / m) * X.T @ d_z_1

    grads = {'d_theta_1': d_theta_1, 'd_theta_2': d_theta_2}

    return grads


def update_network(network, grads, learning_rate=0.02):
    theta_1 = network['theta_1']
    theta_2 = network['theta_2']

    d_theta_1 = grads['d_theta_1']
    d_theta_2 = grads['d_theta_2']

    theta_1 = theta_1 - d_theta_1 * learning_rate
    theta_2 = theta_2 - d_theta_2 * learning_rate

    network = {'theta_1': theta_1,
               'theta_2': theta_2}

    return network


def bp_network(X, Y, n_h=26, n_input=41, n_output=10,
               epochs=10000, trace=False,
               learning_rate=0.0075):
    n_x = X.shape[1]
    n_y = Y.shape[1]

    network = init_network(n_x, n_h, n_y)

    # 梯度下降
    for epoch in range(epochs):
        parameters_trace = \
            forward_propagation(X, network)
        cost = loss_cross_entropy(parameters_trace['a2'], Y)
        grads = backward_propagation(network, parameters_trace,
                                     X, Y)
        network = update_network(network, grads, learning_rate=learning_rate)
        if trace and epoch % 1000 == 0:
            print('迭代第%i次，代价函数为：%f' % (epoch, cost))
            rate = predict(network, X, Y)
            print('准确率为:', rate)

    return network

def predict(network, x_test, y_test):
    theta_1 = network['theta_1']
    theta_2 = network['theta_2']
    X = x_test
    z1 = X @ theta_1
    a1 = sigmoid(z1)
    z2 = a1 @ theta_2
    a2 = sigmoid(z2)

    m, n = y_test.shape
    guess = 0
    for a, b in zip(a2, y_test):
        pre = np.argmax(a)
        real = np.argmax(b)
        if pre == real:
            guess += 1
    return guess / m
if __name__ == "__main__":
    # 数据读取
    X_row = np.array(pd.read_csv('./data/X_data.csv', header=None))
    y_row = pd.DataFrame(pd.read_csv('./data/y_label.csv', header=None))
    ones = np.ones((X_row.shape[0], 1))
    X = np.append(ones, X_row, axis=1)
    Y = np.array(pd.get_dummies(
        y_row,
        columns=[0]
    ))
    shuffle_ix = np.random.permutation(np.arange(len(X)))
    x = X[shuffle_ix]
    y = Y[shuffle_ix]

    row, columns = x.shape
    percent = 0.8
    # 划分训练集和测试集

    x_train = x[:int(percent * row), :]
    y_train = y[:int(percent * row), :]

    x_test = x[int(percent * row):, :]
    y_test = y[int(percent * row):, :]

    # 开始训练
    start_time = time.time()
    network = bp_network(x_train, y_train,
                         n_h=26, n_input=401, n_output=10,
                         epochs=10000, trace=True,
                         learning_rate=0.005
                         )
    end_time = time.time()
    print(end_time - start_time, 's')
    rate = predict(network,x_test,y_test)
    print('准确率为:', rate)
