import numpy as np

x = np.loadtxt('./q4x.dat')
y = np.loadtxt('./q4y.dat', dtype=str).reshape(-1,1)
z = np.zeros(len(y),dtype=float)
mu_0 = np.zeros(2,dtype=float)
mu_1 = np.zeros(2,dtype=float)
x = (x - np.mean(x))/np.std(x)
sigma = np.zeros((2,2),dtype=float)

for i in range(0,len(y)):
    if y[i] == "Alaska":
        z[i] = 0.0
    else:
        z[i] = 1.0
count = 0
tmp_sum1 = 0
tmp_sum2 = 0
tmp_sum3 = 0
tmp_sum4 = 0
n = 2
for i in range(0,len(y)):
    if z[i] == 0:
        tmp_sum1 = tmp_sum1 + x[i, 0]
        tmp_sum2 = tmp_sum2 + x[i, 1]
        count = count + 1
    else:
        tmp_sum3 = tmp_sum3 + x[i, 0]
        tmp_sum4 = tmp_sum4 + x[i, 1]

mu_0[0] = tmp_sum1/count
mu_0[1] = tmp_sum2/count

mu_1[0] = tmp_sum3/(len(y)-count)
mu_1[1] = tmp_sum4/(len(y)-count)

print(mu_0)
print(mu_1)
p = 0
q = 0
r = 0
for i in range(0,len(y)):
    if z[i] == 0:
        p = p + (x[i, 0] - mu_0[0])*(x[i, 0] - mu_0[0])
        q = q + (x[i, 1] - mu_0[1]) * (x[i, 1] - mu_0[1])
        r = r + (x[i, 0] - mu_0[0]) * (x[i, 1] - mu_0[1])
    else:
        p = p + (x[i, 0] - mu_1[0]) * (x[i, 0] - mu_1[0])
        q = q + (x[i, 1] - mu_1[1]) * (x[i, 1] - mu_1[1])
        r = r + (x[i, 0] - mu_1[0]) * (x[i, 1] - mu_1[1])
p = p/len(y)
r = r/len(y)
q = q/len(y)
sigma[0][0] = p
sigma[1][1] = q
sigma[0][1] = r
sigma[1][0] = r
print(sigma)
