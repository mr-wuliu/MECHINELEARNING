import numpy as np
import pandas as pd
import clementine as ct


def sigmoid(x):
    return 1.0/ (1.0+np.exp(-x))


def initialize_network(n_x,n_y,n_nodes,n_layers)->[]:
    """
    神经网络初始化
    :param n_x: 输入节点个数
    :param n_y: 输出节点个数
    :param n_layer: 神经网咯层数
    :param n_noeds: 隐藏层节点个数
    :return: 整个神经网络
    """
    network= [] # 构建神经网络集合
    if n_layers == 0:
        network.append(np.random.randn(n_x, n_y))
        return network

    network.append(np.random.randn(n_x,n_nodes))
    for i in range(n_layers-1):
        network.append(np.random.randn(n_nodes,n_nodes))
    network.append(np.random.randn(n_nodes,n_y))

    return network


def forward_propagation(X, network):
    """
    前向传播, 并保存传播过程数据
    使用Sigmoid作为激活函数
    z 为输入
    alpha 为 输出
    :param X:
    :param network:
    :return:
    """
    n_theta = len(network)  # 权重个数
    z = []
    alpha = []

    z.append(X @network[0])
    alpha.append(sigmoid(z[0]))

    for i in range(1, n_theta):
        z.append(alpha[i-1] @ network[i])
        alpha.append(sigmoid(z[i]))

    return alpha, z


def loss_cross_entropy(alpha, Y, c=True):
    """
    交叉熵损失计算
    :param alpha: 前向传播时保留的传入参数
    :param Y: 期望分类
    :param c: 系数
    :return: 损失值
    """
    m = Y.shape[1]
    alpha_last = alpha[-1]
    # 计算交叉熵
    loss = -np.sum(np.multiply(np.log(alpha_last + 1e-6), Y) + np.multiply((1 - Y), np.log(1 - alpha_last + 1e-6)))
    if c:
        loss =  loss / m
    return loss


def backward_propagation(network, alpha, X, Y):
    """
    反向传播,返回更新梯度
    :param network:
    :param alpha:
    :param X:
    :param Y:
    :return:
    """
    m = Y.shape[1]

    w2 = network['w2']

    a1 = cache['a1']
    a2 = cache['a2']

    # 反向传播，计算dw1、db1、dw2、db2
    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)

    grads = {'dw1': dw1, 'dw2': dw2}

    # 反向传播
    n = len(network)  # 神经网络权重个数
    dz_last = alpha[-1] - Y  # 最后一项
    grad = []
    for i in range()


    return grads


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = ct.load_bp(percent=0.8, stander=False, is_shuffle=True)
