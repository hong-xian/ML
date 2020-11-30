# 单变量线性回归
import numpy as np
import matplotlib.pyplot as plt
print('Plotting Data...')
data = np.loadtxt('/Users/xiujing/Desktop/ML/data_sets/ex1data1.txt', delimiter=',', usecols=(0, 1))
X = data[:, 0]
y = data[:, 1]
m = len(X)
plt.scatter(X, y, marker="o")
plt.axis([0, 25, -5, 25])
plt.xlabel("population")
plt.ylabel("profit")
x_0 = np.zeros((1, m))
X = np.concatenate([x_0, X.reshape(1, m)], axis=0)
y = y.reshape(1, m)


def cost(theta, X, y):
    return (1 / (2 * m)) * np.sum(np.power(np.matmul(X.T, theta) - y.T, 2))


def gradientdescent(theta, alpha, epoch):
    for i in range(epoch):
        descent = (1 / m) * np.matmul(X, (np.matmul(X.T, theta) - y.T))
        theta = theta - alpha * descent
        if i % 10 == 0:
            print(descent)
            print(theta.reshape(2))
            print("%d th cost is %4.f \n" % (i, cost(theta, X, y)))
            f = theta[0, 0] + theta[1, 0] * X
            plt.plot(X, f, "r")


theta = np.zeros((2, 1))
gradientdescent(theta, 0.003, 200)

theta_theo = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X, X.T)), X), y.T)
print(theta_theo)