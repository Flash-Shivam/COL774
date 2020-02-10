import numpy as np
import math

x = np.zeros((3,1000000),dtype=float)

x1 = np.random.randn(1000000)
x1 = (x1 - np.mean(x1))/(np.std(x1))
x1 = np.dot(x1,2)
for i in range(0,1000000):
    x1[i] = x1[i] + 3.0
    x[0, i] = 1
    x[1, i] = x1[i]
# print(x1)

x2 = np.random.randn(1000000)
x2 = (x2 - np.mean(x2))/(np.std(x2))
x2 = np.dot(x2,2)
for i in range(0,1000000):
    x2[i] = x2[i] - 1.0
    x[2, i] = x2[i]


e = np.random.randn(1000000)
e = (e - np.mean(e))/(np.std(e))
e = np.dot(e,math.sqrt(2))

y = np.zeros(1000000,dtype=float)

for i in range(0,1000000):
    y[i] = 3 + x1[i] + 2*x2[i] + e[i]

convergence = False
learning_rate = 0.001
r = 1
theta = np.zeros(3,dtype=float)


def cost(theta,x1,x2,batch,r):
    p = 0
    for i in range(0,r):
        index = (batch - 1) * r + i
        p = p + ((y[index]-(theta[1]*x1[index]+theta[2]*x2[index]+theta[0]))**2)/(2*r)
    return p


end = int(1000000/r)
n_iter = 0
t = 0
tmp = np.zeros(3,dtype=float)
while not convergence:
    for b in range(1, end+1):
        t = 0
        tmp = np.zeros(3, dtype=float)
        for i in range(0,r):
            index = (b - 1) * r + i
            tmp[0] = tmp[0] + learning_rate * (y[index] - (theta[1] * x1[index] + theta[2] * x2[index] + theta[0]))
            tmp[1] = tmp[1] + learning_rate * (x1[index]) * (y[index] - (theta[1] * x1[index] + theta[2] * x2[index] +
                                                                         theta[0]))
            tmp[2] = tmp[2] + learning_rate * (x2[index]) * (y[index] - (theta[1] * x1[index] + theta[2] * x2[index] +
                                                                         theta[0]))
        theta[0] = theta[0] + tmp[0]
        theta[1] = theta[1] + tmp[1]
        theta[2] = theta[2] + tmp[2]
        n_iter = n_iter + 1
        t = t + cost(theta, x1, x2, b, r)
        if abs(tmp[0]) < 0.00000001 and abs(tmp[1]) < 0.00000001 and abs(tmp[2]) < 0.00000001:
            print("Converged", n_iter, theta)
            convergence = not convergence
            break

print(theta, n_iter,r)

convergence = False
r = 100
t = 0
n_iter = 0
theta[0] = 0.0
theta[1] = 0.0
theta[2] = 0.0
end = int(1000000/r)
while not convergence:
    t = 0
    for b in range(1, end+1):
        tmp = np.zeros(3, dtype=float)
        for i in range(0,r):
            index = (b - 1) * r + i
            tmp[0] = tmp[0] + learning_rate *(1/(r))*(y[index] - (theta[1] * x1[index] + theta[2] * x2[index] + theta[0]))
            tmp[1] = tmp[1] + learning_rate *(1/(r))*(x1[index]) * (y[index] - (theta[1] * x1[index] + theta[2] * x2[
                index] +
                                                                         theta[0]))
            tmp[2] = tmp[2] + learning_rate *(1/(r))*(x2[index]) * (y[index] - (theta[1] * x1[index] + theta[2] * x2[
                index] +
                                                                         theta[0]))
        theta[0] = theta[0] + tmp[0]
        theta[1] = theta[1] + tmp[1]
        theta[2] = theta[2] + tmp[2]
        n_iter = n_iter + 1
        if n_iter % 1000 == 0:
            print(theta,tmp[0],tmp[1],tmp[2])
        if abs(tmp[0]) < 0.0001 and abs(tmp[1]) < 0.0001 and abs(tmp[2]) < 0.0001 :
            print("Converged", n_iter, theta)
            convergence = not convergence
            break


print(theta, n_iter, r)

convergence = False
r = 10000
t = 0
n_iter = 0
theta[0] = 0.0
theta[1] = 0.0
theta[2] = 0.0
end = int(1000000/r)
while not convergence:
    for b in range(1, end+1):
        tmp = np.zeros(3, dtype=float)
        for i in range(0,r):
            index = (b - 1) * r + i
            tmp[0] = tmp[0] + learning_rate*(1/(r))*(y[index] - (theta[1] * x1[index] + theta[2] * x2[index] + theta[0]))
            tmp[1] = tmp[1] + learning_rate*(1/(r))*(x1[index]) * (y[index] - (theta[1] * x1[index] + theta[2] * x2[
                index] +
                                                                         theta[0]))
            tmp[2] = tmp[2] + learning_rate * (1/(r))*(x2[index]) * (y[index] - (theta[1] * x1[index] + theta[2] * x2[
                index] +
                                                                         theta[0]))
        theta[0] = theta[0] + tmp[0]
        theta[1] = theta[1] + tmp[1]
        theta[2] = theta[2] + tmp[2]
        # print(theta)
        n_iter = n_iter + 1
        if abs(tmp[0]) < 0.00000001 and abs(tmp[1]) < 0.00000001 and abs(tmp[2]) < 0.00000001:
            print("Converged", n_iter, theta)
            convergence = not convergence
            break

print(theta, n_iter, r)


convergence = False
r = 1000000
n_iter = 0
theta[0] = 0.0
theta[1] = 0.0
theta[2] = 0.0
end = int(1000000/r)
while not convergence:
    for b in range(1, end+1):
        tmp = np.zeros(3, dtype=float)
        for i in range(0,r):
            index = (b - 1) * r + i
            tmp[0] = tmp[0] + learning_rate *(1/(r))* (y[index] - (theta[1] * x1[index] + theta[2] * x2[index] +
                                                                     theta[0]))
            tmp[1] = tmp[1] + learning_rate * (1/(r))*(x1[index]) * (y[index] - (theta[1] * x1[index] + theta[2] * x2[
                index] +
                                                                         theta[0]))
            tmp[2] = tmp[2] + learning_rate * (1/(r))* (x2[index]) * (y[index] - (theta[1] * x1[index] + theta[2] * x2[
                index] +
                                                                         theta[0]))
        theta[0] = theta[0] + tmp[0]
        theta[1] = theta[1] + tmp[1]
        theta[2] = theta[2] + tmp[2]
        n_iter = n_iter + 1
        if abs(tmp[0]) < 0.00000001 and abs(tmp[1]) < 0.00000001 and abs(tmp[2]) < 0.00000001:
            print("Converged", n_iter, theta)
            convergence = not convergence
            break

print(theta, n_iter, r)
