# 多变量线性回归问题
import numpy as np
import matplotlib.pyplot as plt
print('Plotting Data...')
data = np.loadtxt('/Users/xiujing/Desktop/ML/data_sets/ex1data2.txt', delimiter=',')
X1 = data[:, 0].reshape((1, 47))
X2 = data[:, 1].reshape((1, 47))
y = data[:, 2].reshape((47, 1))
m = len(y)
x_0 = np.ones((1, m))
# normalization y.shape=(47, 1) X.shape=(3, 47)
y = y * (1e-5)
X1 = X1 * (1e-3)
X = np.concatenate([x_0, X1, X2], axis=0)


def cost(theta, X, y):
    losssum = np.sum(np.power((np.matmul(X.T, theta) - y), 2))
    return (1 / (2 * m)) * losssum


def Gradientdescent(theta, alpha, epoch):
    for i in range(epoch):
        descent = (1 / m) * np.matmul(X, (np.matmul(X.T, theta) - y))
        theta = theta - alpha * descent
        costlist.append(cost(theta, X, y))
        if i % 200 == 0:
            print("the %d th cost is: %.8f" % (i, cost(theta, X, y)))
            print("and the theta is:", theta.reshape(3), "\n")
    plt.plot(np.arange(epoch), costlist, "r")
# 缺少判定收敛的标准
# theta在变 cost没变？ cost变化极小


costlist = []
theta = np.zeros((3, 1))
Gradientdescent(theta, 0.1, 4000)
theta_theo = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X, X.T)), X), y)
print("the theoritical theta is:\n", theta_theo.reshape(3))
plt.show()