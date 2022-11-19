import numpy as np
import time
import clementine as ct


def bfgs_method(x_train, y_label, max_iter=5000,
                epsilon=1e-4, Lambda=0,
                c=False, log=False):
    """
    实现阻尼牛顿法
    :param x_train:
    :param y_label:
    :param max_iter:
    :param epsilon: 精度
    :param Lambda: 正则化项
    :param c: 损失函数系数
    :param log: 日志选项
    :return:
    """

    m, n = x_train.shape
    theta_k = np.ones((n, 1))
    I = np.eye(n)
    Hk = I
    grad_k = ct.nabla(x_train, theta_k, y_label,
                      c=c, Lambda=Lambda)
    k = 0
    while np.linalg.norm(grad_k) > epsilon and k < max_iter:
        # 牛顿方向
        d = - Hk @ grad_k
        # 搜索步长
        alpha_k = ct.wolfe(x_train, theta_k, y_label, d,
                        c, Lambda)
        # 迭代
        theta_k1 = theta_k + alpha_k * d
        # 更新Hessian
        s_k = theta_k1 - theta_k
        grad_k1 = ct.nabla(x_train, theta_k1, y_label,
                           c=c, Lambda=Lambda)
        y_k = grad_k1 - grad_k
        if s_k.T @ y_k <= 0:
            print("矩阵非正定,求解失败, 退出")
            print(s_k.T @ y_k)
            break
        # BFGS公式
        Hk = ((I - (s_k @ y_k.T) / (s_k.T @ y_k))).T @ Hk @ ((I - (s_k @ y_k.T) / (s_k.T @ y_k))) + ((s_k @ s_k.T) / (s_k.T @ y_k))
        grad_k = grad_k1
        theta_k = theta_k1
        k += 1
        if log and  k % 200 == 0:
            print(k)
    if log and max_iter == k:
        print("迭代超范围,|grad|=",np.linalg.norm(grad_k))
    if log:
        print("损失:", ct.cost(x_train, theta_k, y_label, c=c, Lambda=Lambda))
    return theta_k
