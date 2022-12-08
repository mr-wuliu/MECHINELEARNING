import numpy as np
import pandas as pd
import clementine as ct
import random


class k_meams:
    def __init__(self, x, y, k):
        self.k = k  # 簇数
        self.x = x
        self.y = y
        self.C = [[] for i in range(k)]  # 簇划分
        self.D = self.choice_init(k, self.x)  # 初始化支持向量
        self.N = None  # 过程量, 判断质心是否变化

    def run(self) -> None:
        m, n = self.x.shape
        while self.check(self.N, self.D):

            # 为样本划分类别
            for i in range(m):
                # 在均值向量中选择最近的一个
                cate = self.min_distance(self.x[i], self.D)
                self.C[cate].append(i)  # 只添加下标, 不对数据本身进行操作

            # 计算新的均值向量
            for i in range(self.k):
                vector_new = 0
                for j in self.C[i]:
                    vector_new += self.x[j]
                vector_new *= (1 / len(self.C[i]))
                self.D[i] = vector_new
            self.N = self.D.copy()
    def predict(self):
        pass

    def show(self):
        divide = np.zeros(self.y.shape[0])
        for i in range(len(self.C)):
            for j in range(len(self.C[i])):
                divide[self.C[i][j]] = i
        print(divide)
        return divide

    def cost(self):
        cost = 0
        for i in range(len(self.C)):
            for j in range(len(self.C[i])):
                pass


    @staticmethod
    def check(c1, c2):
        m = len(c2)
        if c1 == None or c2 == None:
            return True
        else:
            flag = 0
            for i in range(m):
                if (c1[i] == c2[i]).all():
                    flag += 1
            if flag >= len(c2):
                return False
            return True
    @staticmethod
    def min_distance(x, D):
        min_item = 0
        min_dis = np.linalg.norm(x-D[0])
        for i in range(1,len(D)):
            distance = np.linalg.norm(x - D[i])
            if distance < min_dis:
                min_item = i
                min_dis = distance
        return min_item

    @staticmethod
    def choice_init(k, x):
        """
        随机抽取k个样本作为初始向量
        :param k:
        :param x:
        :return:
        """
        init_vector = []
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
                item = random.randint(0, m)
            q.add(item)
            init_vector.append(x[item])
        return init_vector


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = ct.load_DRUG1n(percent=0.8)
    p = k_meams(x_train, y_train, y_test.shape[1])
    p.run()
    p.show()






