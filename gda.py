import numpy as np
import math
import matplotlib.pyplot as plt
x = np.loadtxt('./q4x.dat')
y = np.loadtxt('./q4y.dat', dtype=str).reshape(-1,1)
z = np.zeros(len(y),dtype=float)
mu_0 = np.zeros(2,dtype=float)
mu_1 = np.zeros(2,dtype=float)
x = (x - np.mean(x))/np.std(x)
sigma = np.zeros((2,2),dtype=float)
sigma0 = np.zeros((2,2),dtype=float)
sigma1 = np.zeros((2,2),dtype=float)
sig_inv = np.zeros((2,2),dtype=float)

e = []
e1 = []
e3 = []
e2 = []
for i in range(0,len(y)):
    if y[i] == "Alaska":
        z[i] = 0.0
        e.append(x[i, 0])
        e1.append(x[i, 1])
    else:
        z[i] = 1.0
        e2.append(x[i, 0])
        e3.append(x[i, 1])


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
phi = (len(y)-count)/(len(y))
mu_0[0] = tmp_sum1/count
mu_0[1] = tmp_sum2/count

mu_1[0] = tmp_sum3/(len(y)-count)
mu_1[1] = tmp_sum4/(len(y)-count)

print(mu_0)
print(mu_1)
p = 0
q = 0
r = 0
p1 = 0
count = 0
q1 = 0
r1 = 0
for i in range(0,len(y)):
    if z[i] == 0:
        p = p + (x[i, 0] - mu_0[0])*(x[i, 0] - mu_0[0])
        q = q + (x[i, 1] - mu_0[1]) * (x[i, 1] - mu_0[1])
        r = r + (x[i, 0] - mu_0[0]) * (x[i, 1] - mu_0[1])
        count = count + 1
    else:
        p1 = p1 + (x[i, 0] - mu_1[0]) * (x[i, 0] - mu_1[0])
        q1 = q1 + (x[i, 1] - mu_1[1]) * (x[i, 1] - mu_1[1])
        r1 = r1 + (x[i, 0] - mu_1[0]) * (x[i, 1] - mu_1[1])


sigma[0][0] = (p+p1)/len(y)
sigma[1][1] = (q+q1)/len(y)
sigma[0][1] = (r+r1)/len(y)
sigma[1][0] = (r+r1)/len(y)
print(sigma)
sigma0[0][0] = p/(len(y)-count)
sigma0[1][1] = q/(len(y)-count)
sigma0[0][1] = r/(len(y)-count)
sigma0[1][0] = r/(len(y)-count)
print(sigma0)
sigma1[0][0] = p1/(len(y)-count)
sigma1[1][1] = q1/(len(y)-count)
sigma1[0][1] = r1/(len(y)-count)
sigma1[1][0] = r1/(len(y)-count)
print(sigma1)


sig_inv = np.linalg.pinv(sigma)
x1 = np.linspace(-2,0,80)

x2 = ((mu_1[0]-mu_0[0])*sig_inv[0][0]*(2*x1-mu_0[0]-mu_1[0])+sig_inv[1][1]*(mu_0[1]-mu_1[1])*(mu_0[1]+mu_1[1])+(sig_inv[0][1]+sig_inv[1][0])*(x1*(mu_1[1]-mu_0[1])+(mu_0[0]*mu_0[1]-mu_1[0]*mu_1[1]))+2*math.log(phi/(1-phi)))/(2*sig_inv[1][1]*(mu_0[1]-mu_1[1])+(sig_inv[0][1]+sig_inv[1][0])*(mu_0[0]-mu_1[0]))

plt.scatter(e, e1, label= "Alaska", color= "blue", marker= "+", s=30)
plt.scatter(e2, e3, label= "Canada", color= "red", marker= "*", s=30)
plt.plot(x1, x2, '-g', label='Decision Boundary')
plt.xlabel('Fresh Water')
plt.grid()
plt.ylabel('Marine Water')
plt.legend()
plt.show()



