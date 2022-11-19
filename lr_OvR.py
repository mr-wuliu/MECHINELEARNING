import time

import numpy as np
import BFGS as bf
import L_BFGS as lbf
import clementine as ct


if __name__ == '__main__':
    # 标记每一个分类的起止
    start = []
    X, y = ct.load_cancer_amazon(filename="cancer", stander=True)
    tags = {}
    for i in range(len(y)):
        if str(y[i]) not in tags:
            tags[str(y[i])] = 0
            start.append(i)
    start.append(len(X))
    # 采用OVR的方法
    # 一共有len(tags)类的数据集合

    theta = []  # 存放theta
    for i in range(len(tags)):
        time_start = time.time()
        y_tr = np.zeros((len(X), 1))
        y_tr[start[i]:start[i + 1]] = 1
        theta.append(lbf.l_bfgs_method(X, y_tr, epsilon=1e-8, c=False, Lambda=0.2, log=True))
        time_end = time.time()
        print("分类:",i,"花费时间:", time_end - time_start)
    pre = np.zeros((len(y), 1))
    guess = 0

    for i in range(len(y)):

        for k in range(len(tags)):
            activation = ct.sigmoid(X[i] @ theta[k])
            if activation > 0.5:
                pre[i] = k + 1
        if pre[i] == y[i]:
            guess += 1
        else:
            print(pre[i], y[i])  # 只显示预测失败样本
    print("预测准确率", guess / len(y))
