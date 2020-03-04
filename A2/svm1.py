import csv
from cvxopt import matrix
from cvxopt import solvers
import numpy as np
import math

from sklearn.svm import SVC
# svclassifier = SVC(kernel='linear')
svclassifier = SVC(kernel="rbf", gamma=0.05)

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

# print(n,m)

z = 0
for i in range(0,len(rows)):
    if int(float(rows[i][784])) == 8 or int(float(rows[i][784])) == 9:
        y[z] = float(int(float(rows[i][784])) - 8)
        for j in range(0,n):
            x[z, j] = float(rows[i][j])/float(255)
        z = z + 1


# print(z)

svclassifier.fit(x,y)


filename = "test.csv"
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
for i in range(0,len(rows1)):
    if int(float(rows1[i][n])) == 8 or int(float(rows1[i][n])) == 9:
        # print(int(float(rows1[i][n])))
        y1[z] = float(int(float(rows1[i][n])) - 8)
        for j in range(0,n):
            x1[z, j] = float(rows1[i][j])/float(255)
        z = z + 1



# print(z)

y2 = svclassifier.predict(x1)

accuracy = (y1 == y2).sum()/float(m)

print(accuracy)
