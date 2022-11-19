import numpy as np
import time
import clementine as ct

np.random.seed(3)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def d_tanh(x):
    return 1 - np.power(np.tanh(x), 2)


def init_network(n_x, n_h, n_y):
    theta_1 = np.random.randn(n_x, n_h) * 0.01
    theta_2 = np.random.randn(n_h, n_y) * 0.01

    # t1 = pd.read_csv("./data/Theta1.csv",header=None)
    # t2 = pd.read_csv("./data/Theta2.csv",header=None)
    # theta_1 = np.array(t1)
    # ones = np.ones((1,401))
    # theta_1 = np.append(ones, theta_1, axis=0).T
    # theta_2 = np.array(t2).T
    network_2 = {'theta_1': theta_1, 'theta_2': theta_2}

    return network_2


def forward_propagation(X, network_l):
    theta_1 = network_l['theta_1']
    theta_2 = network_l['theta_2']
    z1 = X @ theta_1
    a1 = np.tanh(z1)
    z2 = a1 @ theta_2
    a2 = sigmoid(z2)
    trace = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return trace


def backward_propagation(network_b, trace, X, Y):
    m = Y.shape[0]
    theta_2 = network_b['theta_2']
    a_1 = trace['a1']
    a_2 = trace['a2']

    d_z_2 = a_2 - Y
    d_theta_2 = (1 / m) * (a_1.T @ d_z_2)
    # d_z_1 = np.multiply(d_z_2 @ theta_2.T, d_sigmoid(a_1))
    d_z_1 = np.multiply(d_z_2 @ theta_2.T, d_tanh(a_1))
    d_theta_1 = (1 / m) * X.T @ d_z_1
    grads = {'d_theta_1': d_theta_1, 'd_theta_2': d_theta_2}

    return grads


def update_network(network_u, grads, learning_rate=0.02):
    theta_1 = network_u['theta_1']
    theta_2 = network_u['theta_2']

    d_theta_1 = grads['d_theta_1']
    d_theta_2 = grads['d_theta_2']

    theta_1 = theta_1 - d_theta_1 * learning_rate
    theta_2 = theta_2 - d_theta_2 * learning_rate

    network_u = {'theta_1': theta_1, 'theta_2': theta_2}

    return network_u


def train_dnn(X, Y, n_h,
              epochs=10000,
              print_cost=True, learning_rate=0.01):
    global cost
    n_x = X.shape[1]
    n_y = Y.shape[1]
    network_1 = init_network(n_x, n_h, n_y)
    # 梯度下降循环
    for i in range(0, epochs):
        trace = forward_propagation(X, network_1)
        cost = cross_entropy_loss(trace['a2'], Y)
        grads = backward_propagation(network_1, trace, X, Y)
        network_1 = update_network(network_1, grads, learning_rate=learning_rate)
        if print_cost and i % 1000 == 0:
            print('迭代第%i次，代价函数为：%f' % (i, cost), end="")
            print("  测试集预测准确率:", predict(network_1, x_test, y_test), end="")
            rate = predict(network_1, x_train, y_train)
            print("  数据集预测准确率:", rate)
            data = ct.shuffle(X,Y)
            X = data[0]
            Y = data[1]
    return network_1


def cross_entropy_loss(Pre_y, Y):
    m = Y.shape[0]
    loss = np.multiply(np.log((Pre_y + 1e-6)), Y) + np.multiply((1 - Y), np.log(1 - Pre_y + 1e-6))
    cost = - np.sum(loss) / m
    return cost


def predict(network_p, x_test, y_test):
    theta_1 = network_p['theta_1']
    theta_2 = network_p['theta_2']

    z1 = np.dot(x_test, theta_1)
    a1 = np.tanh(z1)
    # a1 = np.tanh(z1)
    z2 = np.dot(a1, theta_2)
    a2 = sigmoid(z2)

    rows, columns = y_test.shape
    guess = 0
    for a, b in zip(a2, y_test):
        pre = np.argmax(a)
        real = np.argmax(b)

        if pre == real:
            guess += 1
    return guess / rows


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = ct.load_bp(0.8)

    start_time = time.time()
    network = train_dnn(x_train, y_train, 26,
                        epochs=2000000, learning_rate=0.02)
    end_time = time.time()
    print(end_time - start_time, 's')
    rate = predict(network, x_test, y_test)
    print('准确率为:', rate)
