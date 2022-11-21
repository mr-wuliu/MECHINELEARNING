from sklearn import svm
import clementine as ct
import numpy as np


if __name__ == '__main__':

    np.random.seed(1)
    # 导入数据集并对训练集打乱
    x_train, y_train, x_test, y_test = ct.load_telco(percent=0.8, stander=True,is_shuffle=True)


    # 使用SVM
    clf = svm.SVC(C=2.0, kernel='rbf', gamma=0.5)
    clf.fit(x_train, np.ravel(y_train))

    guess = 0
    y_reg = clf.predict(x_train)
    for a, b in zip(y_reg, y_train):
        if a == b:
            guess += 1

    print("回归测试:",guess / y_train.shape[0] * 100,'%')

    guess= 0
    y_pre = clf.predict(x_test)
    for a, b in zip(y_pre, y_test):
        if a == b:
            guess += 1

    print("测试集准确率为:", guess/ y_test.shape[0]*100,'%')


