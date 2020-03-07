import csv
from cvxopt import matrix
from cvxopt import solvers
from sklearn import svm
import numpy as np
import math
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
# svclassifier = SVC(kernel='linear')
clf = svm.SVC(decision_function_shape='ovo')

filename = "train.csv"
rows = []


with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)
print("Rows appended")
n = len(rows[0])-1
m = len(rows)
x = np.zeros((m,n),dtype = 'd')
y = np.zeros(m,dtype='d')

# print(n,m)

z = 0
for i in range(0,len(rows)):
    y[z] = (int(float(rows[i][784])))
    for j in range(0,n):
        x[z, j] = float(rows[i][j])/float(255)
    z = z + 1

print("Feautes Selected")
# print(z)

clf.fit(x,y)

print("Fitted")
filename = "test.csv"
rows1 = []



with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows1.append(row)

print("RR2")
n = len(rows1[0])-1
m = len(rows1)
x1 = np.zeros((m,n),dtype = 'd')
y1 = np.zeros(m,dtype='d')

# print(n,m)

z = 0
for i in range(0,len(rows1)):
    y1[z] = int(float(rows1[i][784]))
    for j in range(0,n):
        x1[z, j] = float(rows1[i][j])/float(255)
    z = z + 1


print("FF2")
# print(z)

y2 = clf.predict(x1)

print("Predicted")

accuracy = (y1 == y2).sum()/float(m)

print(accuracy)

y_True = []
y_pred = []
for i in range(0,m):
    y_True.append(int(y1[i]))
    y_pred.append(int(y2[i]))

confusionmatrix = confusion_matrix(y_True,y_pred)
print(confusionmatrix)
df_cm = pd.DataFrame(confusionmatrix, index = ['0','1','2','3','4','5','6','7','8','9'],
                  columns =  ['0','1','2','3','4','5','6','7','8','9'])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()
