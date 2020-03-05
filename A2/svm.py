import csv
from cvxopt import matrix
from cvxopt import solvers
import numpy as np
import math

filename = "train.csv"
rows = []

pos = 0
neg = 0


with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)
        l = len(row)
        index = int(float(row[l-1]))
        if index == 8:
            pos = pos + 1
        elif index == 9:
            neg = neg + 1


n = len(rows[0])-1
m = pos + neg
x = np.zeros((m,n),dtype = 'd')
y = np.zeros(m,dtype='d')

print(n,m)

z = 0
for i in range(0,len(rows)):
    if int(float(rows[i][784])) == 8 or int(float(rows[i][784])) == 9:
        y[z] = (int(float(rows[i][784])) - 8)
        if y[z] == 0:
            y[z] = -1.0
        for j in range(0,n):
            x[z, j] = float(rows[i][j])/float(255)
        z = z + 1

print(y)
print(z)
temp = np.zeros((m,m),dtype='d')
temp1 = np.zeros((2*m,1),dtype='d')
temp2 = np.zeros((2*m,m),dtype='d')
temp3 = np.zeros((1,m),dtype='d')
for i in range(0,m):
    temp1[i, 0] = 1
    temp2[i, i] = 1
    temp2[m+i, i]= -1
    temp3[0, i] = y[i]

gamma = 0.05
for i in range(0,m):
    for j in range(0,m):
        # res = math.exp(np.sum(np.square(np.add(x[i],-x[j])))*-gamma)
        res1 = np.dot(x[i],x[j])
        # print(res,res1)
        temp[i, j] = y[i]*y[j]*(res1)


h = matrix((temp1))

G = matrix(temp2)

b = matrix(np.array([0.0]))
q =  matrix(np.ones((m,1),dtype = 'd')*-1)
A = matrix((temp3))
P = matrix(temp)

w = np.zeros((1,n),dtype='d')
sol = solvers.qp(P,q,G,h,A,b)
#print(sol['x'])
s = 0
num = 0
for i in sol['x']:
    if not (abs(i) < 0.0001):
        num = num + 1
    alpha = i*y[s]
    w = np.add(w,np.dot(x[s],alpha))
    s = s + 1
#print(d)
print(num,m)

r = 1000000000
s = -1
for i in range(0,m):
    p = np.dot(w,x[i])
    if y[i] == 1:
        if p < r:
            r = p
    else:
        if p> s:
            s = p

b = -(s+r)/2
print(r,s)



print("Total NUmber of SVM: " + str(num))


filename = "val.csv"
rows1 = []

pos = 0
neg = 0


with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows1.append(row)
        l = len(row)
        index = int(float(row[l-1]))
        if index == 8:
            pos = pos + 1
        elif index == 9:
            neg = neg + 1


n = len(rows1[0])-1
m = pos + neg
x1 = np.zeros((m,n),dtype = 'd')
y1 = np.zeros(m,dtype='d')

# print(n,m)

z = 0
y2 = np.zeros(m,dtype='d')
for i in range(0,len(rows1)):
    if int(float(rows1[i][n])) == 8 or int(float(rows1[i][n])) == 9:
         # print(int(float(rows1[i][n])))
        y1[z] = (int(float(rows1[i][n])) - 8)
        if y1[z] == 0:
            y1[z] = -1.0
        for j in range(0,n):
            x1[z, j] = float(rows1[i][j])/float(255)
        resf = np.dot(w,x1[z]) + b

        if resf < 0:
            y2[z] = -1
        else:
            y2[z] = 1
        z = z + 1

accuracy = (y1 == y2).sum()/float(m)

print(accuracy)
