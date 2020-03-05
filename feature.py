import csv
import math
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def convert_bigram(word_list):
    l = len(word_list)
    if l == 0 or l == 1:
        return word_list
    else:
        res = word_list
        for i in range(0,l-1):
            res.append(word_list[i] + " " + word_list[i+1])
        return res
# training.1600000.processed.noemoticon.csv
x = []
filename = "training.1600000.processed.noemoticon.csv"
rows = []
cond = 0
with open(filename, 'r', encoding='ISO-8859-1') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)


m = len(rows)

y = np.zeros(m,dtype=float)
count = 0
# print("Rows Appended")
di0 = dict()
di1 = dict()
# di2 = dict()
count1 = 0
count2 = 0
z = 0
vocab = []
for i in rows:
    p = (i[5]).split()
    p = convert_bigram(p)
    # print(p)
    x.append(p)

    if i[0] == "4":
        for j in p:
            if j in di0:
                di0[j] = di0[j] + 1
            else:
                di0[j] = 1
            vocab.append(j)
        y[z] = 1.0
        z = z + 1
        count = count + 1
        count1 = count1 + len(p)
    else:
        for j in p:
            if j in di1:
                di1[j] = di1[j] + 1
            else:
                di1[j] = 1
            vocab.append(j)
        y[z] = 0.0
        z = z + 1
        count2 = count2 + len(p)


# print(len(di0),len(di1),len(set(vocab)),z)

v = len(set(vocab))
count1 = count1 + v
count2 = count2 + v

phi = float(count)/float(m)
# print(phi,v)


t = 0
filename1 = "testdata.manual.2009.06.14.csv"
rows1 = []
with open(filename1, 'r', encoding='ISO-8859-1') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if row[0] == "4" or row[0] == "0":
            t = t + 1
            rows1.append(row)

# print(t)

y1 = np.zeros(t,dtype=float)
c = 0
x1 = []
for i in rows1:
    p = (i[5]).split()
    p = convert_bigram(p)
    x1.append(p)
    if i[0] == "4":
        y1[c] = 1.0
    elif i[0] == "0":
        y1[c] = 0.0
    c = c+1

# print(c,len(x1))
q = 0.0
res1 = np.zeros(t,dtype=float)
for i in range(0,t):
    n1 = len(x1[i])
    sum1 = 0.0
    sum2 = 0.0
    for j in range(0,n1):
        sum1 = sum1 + math.log(di0.get(x1[i][j],0)+1) - math.log(count1)
        sum2 = sum2 + math.log(di1.get(x1[i][j],0)+1) - math.log(count2)
    sum1 = sum1 + math.log(phi)
    sum2 = sum2 + math.log(1-phi)

    val = max(sum1,sum2)
    if val == sum1:
        res1[i] = 1
    else:
        res1[i] = 0
    if y1[i] == res1[i]:
        q = q + 1

print("Accuracy Over Test Data :",float(100*q)/float(t))
