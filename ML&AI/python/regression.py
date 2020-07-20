# %%
import csv
import numpy as np
import random as random
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

x_data = [338.,  333.,  328., 207., 226., 25., 179.,  60., 208.,  606.]
y_data = [640., 633.,  619., 393., 428.,   27., 193.,  66.,  226., 1591.]


def getGrad(b, w):
    b_grad = 0.0
    w_grad = 0.0
    for i in range(10):
        b_grad += (-2.0)*(y_data[i]-(b+w*x_data[i]))
        w_grad += (-2.0*x_data[i])*(y_data[i]-(b+w*x_data[i]))
    return (b_grad, w_grad)


bias = np.arange(-200, -100, 1)
weight = np.arange(-5, 5, 0.1)
Z = np.zeros((len(bias), len(weight)))  # color

for i in range(len(bias)):
    for j in range(len(weight)):
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] += (y_data[n] - (bias[i] + weight[j] * x_data[n])) ** 2
        Z[j][i] /= len(x_data)

# y_data = b + w * x_data
b = -120  # initial b
w = -4  # initial w
lr = 1  # learning rate
iteration = 100000
b_history = [b]
w_history = [w]

lr_b = 0
lr_w = 0

for i in range(iteration):
    b_grad, w_grad = getGrad(b, w)

    lr_b += b_grad ** 2
    lr_w += w_grad ** 2

    b -= lr/np.sqrt(lr_b) * b_grad
    w -= lr/np.sqrt(lr_w) * w_grad

    b_history.append(b)
    w_history.append(w)

# plot the figure
plt.contourf(bias, weight, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()

# %%
