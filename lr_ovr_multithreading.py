import random
from threading import Thread
import numpy as np
import BFGS as bf
import clementine as ct
import time


from threading import Thread
import time
import random
'''任务场景
一个完整的流程，需要A函数循环执行
已知：A的处理时间是动态的，线程数固定
原来的执行逻辑：拥有一定数目的线程
1.所有的线程，全部执行A函数，然后join()，等待所有线程结束，
2.重复执行1
优化：为了提高单位时间内处理数据量，现在组织提出要求提高处理效率
可优化点：
1.全部线程执行A函数时，每个线程执行的时间是不固定的，当一个线程执行时间较长，但其他线程较短已完成时
2.由于线程join()的阻塞作用，存在多个线程等待一个线程的情况，所有全部完成后，再执行下一步操作，这样浪费了资源
优化猜想：在线程数一定时，分配空闲线程给A函数，减少线程空闲时间，使其满负荷

'''


if __name__ == '__main__':

    # 标记每一个分类的起止
    start = []
    X, y = ct.load_cancer_amazon(filename="amazon", stander=True)
    tags = {}
    for i in range(len(X)):
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
        theta.append(bf.bfgs_method(X, y_tr, epsilon=1e-6,max_iter=5, c=True, Lambda=0, log=True))
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
