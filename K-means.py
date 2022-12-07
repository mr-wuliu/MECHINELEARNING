import numpy as np
import pandas as pd
import clementine as ct
import random

def L2(x1, x2):
    return np.linalg.norm(x1-x2)

class k_meams:
    def __init__(self, x, y, k):
        self.k = k  # 簇数
        self.x = x
        self.y = y
        self.C = [[] for i in range(k)]  # 簇划分
        self.D = self.choice_init(k, self.x)  # 初始向量
    def run(self):
        m, n = self.x.shape
        for i in range(m):
            





    @staticmethod
    def choice_init(k, x):
        """
        随机抽取k个样本作为初始向量
        :param k:
        :param x:
        :return:
        """
        inti_vector = []
        q = set()
        m, n = x.shape
        for i in range(k):
            item = random.randint(0, m)
            epoch = 0
            while item in q:
                epoch += 1
                if epoch >= 5:
                    print("连续5次未抽中新样本, 退出")
                    return None
                item = random.randint(0,m)
            q.add(item)
            inti_vector.append(x[item])
        return inti_vector

    def show(self):
        print(self.D[0])


if __name__ == '__main__':
    test = np.array([1,2,3])
    p = k_meams(test,test,3)
    p.show()





