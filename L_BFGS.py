import numpy as np
import clementine as ct


def two_loop(s, y, rho, gk,col):
    """
    通过序列s , y rho 序列计算出d_k方向
    :param gk: 当然梯度
    :param s: 字典, 点差集合
    :param y: 字典, 梯度差集合
    :param rho: 字典, 标量集合
    :return: d_k, 牛顿方向
    """
    n = len(s)  # 计算列表长度

    # 选择近似矩阵

    a = np.ones((n, 1))  # a[i] 同样为标量
    q = gk.copy()  # n * 1 矩阵

    # 第一次循环
    for i in range(n - 1, -1, -1):
        a[i] = rho[i] @ s[i].T @ q
        q -= a[i] * y[i]

    # 选择近似矩阵
    h0 = np.eye(col)
    if n >= 1:
        h0 = (s[-1].T @ y[-1]) / (y[-1].T @ y[-1]) * h0
    # 第二次循环
    d = h0 @ q

    for i in range(n):
        b = rho[i] @ y[i].T @ d
        d += s[i] @ (a[i] - b)
    return d


def l_bfgs_method(x_train, y_label, max_iter=5000,
                  epsilon=1e-4, Lambda=0, m=5,
                  c=False, log=False):
    # 参数初始化
    row, col = x_train.shape
    theta_k = np.zeros((col, 1))  # 初始位置
    grad_k = ct.nabla(x_train, theta_k, y_label,
                      Lambda=Lambda, c=c)
    k = 0
    s, y, rho = [], [], []

    while np.linalg.norm(grad_k) > epsilon and k < max_iter:
        d_k = -two_loop(s, y, rho, grad_k,col)

        alpha = ct.wolfe(x_train, theta_k, y_label, d_k,
                      c=c, Lambda=Lambda)
        theta_k1 = theta_k + alpha * d_k

        s_k = theta_k1 - theta_k
        grad_k1 = ct.nabla(x_train, theta_k1, y_label,
                           Lambda=Lambda, c=c)
        y_k = grad_k1 - grad_k

        if s_k.T @ y_k <= 0:
            print("矩阵非正定,求解失败, 退出")
            print(s_k.T @ y_k)
            break
        rho.append(1 / (s_k.T @ y_k))
        s.append(s_k)
        y.append(y_k)
        if (len(rho) > m):
            rho.pop(0)
            s.pop(0)
            y.pop(0)
        # 更新
        k += 1
        if log and k % 10 == 0:
            print("迭代次数",k,"||grad||",np.linalg.norm(grad_k))
            print("loss", ct.cost(x_train, theta_k1, y_label,
                                  Lambda=Lambda, c=c))
        grad_k = grad_k1
        theta_k = theta_k1
    # end while

    return theta_k
