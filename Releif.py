import numpy as np
import random
import clementine as ct
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


def distance_attribute(a, b):
    """
    定义样本间距离
    如果为字符串, 那么判断其是否相等, 相等距离为零, 否则为1
    如果为数字, 那么相减, 返回差的绝对值
    :param x:
    :return:
    """
    if type(a) == str:
        if a == b:
            return 0
        else:
            return 1
    return abs(a-b)

def distance_sample(a, b):
    dis = 0
    for x, y in zip(a, b):
        dis += distance_attribute(x,y)
    return dis


def releif(x, y,epochs=100):
    """
    :param epochs:
    :param x:
    :param y:
    :return:
    """

    # 区分正例和反例
    row, columns = x.shape
    class_one = []  # 正例
    class_two = []  # 反例
    for i in range(y.shape[0]):
        if str(y.iloc[i]) == '1':
            class_one.append(x.iloc[i])
        elif str(y.iloc[i]) == '2':
            class_two.append(x.iloc[i])
        else:
            print("ERROR")
    # 建立权重
    theta = np.zeros(columns)
    # 随机抽样一百轮
    for i in range(epochs):
        item = random.randint(0, row-1)
        label = y.iloc[item]
        sample = x.iloc[item]
        # 最近的正例
        item_one = 0
        dist_one = 10000
        item_two = 0
        dist_two = 10000

        # 第一类选择最近邻
        for j in range(len(class_one)):
            if distance_sample(sample, class_one[j]) < dist_one:
                dist_one = distance_sample(sample, class_one[j])
                item_one = j
        # 第二类选择最近邻

        for j in range(len(class_two)):
            if distance_sample(sample, class_two[j]) < dist_two:
                dist_two = distance_sample(sample, class_one[j])
                item_two = j
        near_one = class_one[item_one]
        near_two = class_two[item_two]

        for k in range(20):
            if str(label) == '1':
                theta[k] = theta[k] \
                           + distance_attribute(sample[k], near_one[k]) \
                           - distance_attribute(sample[k], near_two[k])
            elif str(label) == '2':
                theta[k] = theta[k] \
                           - distance_attribute(sample[k], near_one[k]) \
                           + distance_attribute(sample[k], near_two[k])
            else:
                print("ERROR2")
    theta = np.argsort(theta)
    return theta[0: theta.shape[0]//2+1]


def logistic_feature_selection(X,y):

    X_scaler = X
    lr_clf = LogisticRegression(solver='saga', max_iter=10000)
    # 以模型系数的均值和中位数构建筛选器
    selector_median = SelectFromModel(estimator=lr_clf, threshold='median')
    selector_median.fit_transform(X_scaler, y)

    return selector_median.get_support(indices=True)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = ct.load_german_clean(percent=0.7,stander=True,is_shuffle=True)
    theta_1 = releif(x_train, y_train,epochs=100)

    theta_2 = logistic_feature_selection(x_train, y_train)

    print(len(theta_1))
    print(len(theta_2))
    for i in theta_2:
        print(x_train.columns[i])



