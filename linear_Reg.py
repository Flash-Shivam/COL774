import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import csv
import plotly.graph_objects as go
import seaborn as sns
start_time = time.time()
n = 1
m = 100

filename = "linearX.csv"
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
e = []
e1 = []
x = np.zeros((m, n+1), dtype=float)


for i in range(0, len(rows)):
    x[i][1] = float(rows[i][0])

filename = "linearY.csv"
fields1 = []
rows1 = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields1 = next(csvreader)
    for row in csvreader:
        rows1.append(row)

for i in range(0, len(rows1)):
    y[i] = rows1[i][0]
    e.append(y[i])

condition = False

for i in range(0,m):
    x[i, 0] = 0

tmp1 = np.ones(m, dtype=float)
for i in range(0, m):
    tmp1[i] = x[i, 1]

# Normalisation
mean = np.mean(tmp1)
variance = (np.std(tmp1))

r = []
for i in range(0, m):
    x[i, 1] = (x[i, 1] - mean)/variance
    r.append(x[i, 1])

'''
for i in range(0, m):
    mean = np.mean(x[i])
    variance = math.sqrt(np.cov(x[i]))
    # print(mean,variance,i,x[i][1])
    for j in range(0, n+1):
        x[i, j] = (x[i, j] - mean)/variance
'''
for i in range(0,m):
    x[i, 0] = 1


def costfunc(theta, y, m, n):
    val = 0.0
    for i in range(0,m):
        val1 = hypo(theta, x, i, n)
        val1 = val1 - y[i]
        val1 = val1*val1
        val = val + val1
    val = val/(2)
    return val


def hypo(theta, x, index, n):
    value = 0
    for i in range(0, n+1):
        value = value + theta[i]*x[index, i]
    return value


n_iter = 0
v1 = []
v2 = []
v3 = []


while not convergence:
    for i in range(0, n+1):
        val = 0
        for j in range(0, m):
            val = val + (hypo(theta, x, j, n)-y[j])*x[j, i]
        val = val*learning_rate
        tmp[i] = theta[i] - val
    for i in range(0, n+1):
        theta[i] = tmp[i]
    error = costfunc(theta, y, m, n)
    n_iter = n_iter + 1
    v1.append(theta[0])
    v2.append(theta[1])
    v3.append(error)
    # time.sleep(0.1)
    if error < 0.0001 or n_iter > 1000:
        convergence = not convergence
    # print(error)
# print(r)
print(theta)

x1 = np.linspace(-2,5,70)
plt.scatter(r, e, label="Wine", color= "blue", marker= "+", s=30)
y1 = theta[1]*x1 + theta[0]
plt.plot(x1, y1, '-g', label='Hypothesis Function')
plt.xlabel('Acidity of wine')
plt.grid()
plt.ylabel('Density of Wine')
plt.legend()
plt.show()

xs = np.linspace(0,2,20)
ys = np.linspace(-1,1,20)
xs, ys = np.meshgrid(xs,ys)
p = 0
for i in range(0, m):
    p = p + (y[i]-(xs + ys*x[i, 1]))*(y[i]-(xs + ys*x[i, 1]))

p = p/2
zs = p

fig = plt.figure(figsize=(7,7))
ax = fig.gca()
plt.contour(xs, ys, zs,cmap='viridis')
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
for i in range(0,len(v1)):
    f1 = []
    f2 = []
    f1.append(v1[i])
    f2.append(v2[i])
    ax.plot(f1, f2, color='orange', marker='.')
    fig.tight_layout()
    fig.canvas.draw()
    time.sleep(0.005)
    # print("Doing", i)

plt.show()


