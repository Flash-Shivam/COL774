import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import seaborn as sns
n = 2
m = 100

filename = "logisticX.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

convergence = False

theta = np.zeros((n+1), dtype=float)

learning_rate = 0.0015

tmp = np.zeros(n+1,dtype=float)

y = np.zeros(m, dtype=float)

x = np.zeros((m, n+1), dtype=float)

h = np.zeros((n+1,n+1),dtype=float)

h_inv = np.zeros((n+1,n+1),dtype=float)

grad = np.zeros(n+1,dtype=float)
for i in range(0, len(rows)):
    x[i][1] = float(rows[i][0])
    x[i][2] = float(rows[i][1])

filename = "logisticY.csv"
fields1 = []
rows1 = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields1 = next(csvreader)
    for row in csvreader:
        rows1.append(row)

for i in range(0, len(rows1)):
    y[i] = rows1[i][0]

condition = False

for i in range(0,m):
    x[i, 0] = 0


tmp1 = np.ones(m, dtype=float)
tmp2 = np.ones(m, dtype=float)
for i in range(0, m):
    tmp1[i] = x[i, 1]
    tmp2[i] = x[i, 2]
# Normalisation
mean = np.mean(tmp1)
variance = (np.std(tmp1))

r = []
r1 = []
for i in range(0, m):
    x[i, 1] = (x[i, 1] - mean)/variance
    if y[i] == 0:
        r.append(x[i, 1])
    else:
        r1.append(x[i, 1])

mean = np.mean(tmp2)
variance = (np.std(tmp2))

r2 = []
r3 = []
for i in range(0, m):
    x[i, 2] = (x[i, 2] - mean)/variance
    if y[i] == 0:
        r2.append(x[i, 2])
    else:
        r3.append(x[i, 2])

for i in range(0,m):
    x[i, 0] = 1


def sigmoid(x):
    # print("x = ", x)
    val = 1/(1+math.exp(-x))
    return val


def hypo(theta, x, index, n):
    sum = 0
    for i in range(0,n+1):
        sum = sum + theta[i]*x[index, i]
    return sigmoid(sum)


def cost(theta,y,x,m,n):
    p = 0
    for i in range(0,m):
        p = p + y[i]*math.log(hypo(theta,x,i,n)) + (1-y[i])*math.log(1-hypo(theta,x,i,n))
    return -p


n_iter = 0

while not convergence:

    for i in range(0,n+1):
        for j in range(0,n+1):
            res = 0
            for k in range(0,m):
                res = res - x[k, i]*x[k, j]*hypo(theta,x,k,n)*(1-hypo(theta,x,k,n))
            h[i, j] = res
    h_inv = np.linalg.inv(h)
    # print(h)
    for i in range(0,n+1):
        res = 0
        for k in range(0,m):
            res = res + x[k, i]*(y[k]-hypo(theta,x,k,n))
        grad[i] = res
    for i in range(0,n+1):
        res = 0
        for k in range(0,n+1):
            res = res + h_inv[i, k]*grad[k]
        theta[i] = theta[i] - res
    n_iter = n_iter + 1
    if n_iter > 10 or cost(theta,y,x,m,n) < 0.0001:
        convergence = not convergence
    # print(cost(theta,y,x,m,n), n_iter)
print(theta)

plt.scatter(r, r2, label="Y1", color= "blue", marker= "+", s=30)
plt.scatter(r1, r3, label="Y2", color= "red", marker= "*", s=30)
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
x1 = np.linspace(-3,3,80)
y1 = -(theta[0]+theta[1]*x1)/(theta[2])
plt.plot(x1, y1, '-g', label='Decision Boundary')
plt.grid()
plt.show()



