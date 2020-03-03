import csv
import math
import numpy as np


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
print("Rows Appended")
di0 = dict()
di1 = dict()
di2 = dict()
count1 = 0
count2 = 0
z = 0
vocab = []
for i in rows:
    p = (i[5]).split()
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


print(len(di0),len(di1),len(set(vocab)),z)

v = len(set(vocab))
count1 = count1 + v
count2 = count2 + v

phi = float(count)/float(m)
print(phi,v)


res = np.zeros(m,dtype=float)
q = 0.0

for i in range(0,m):
    n = len(x[i])
    sum1 = 0.0
    sum2 = 0.0
    for j in range(0,n):
        sum1 = sum1 + math.log(di0.get(x[i][j],0)+1) - math.log(count1)
        sum2 = sum2 + math.log(di1.get(x[i][j],0)+1) - math.log(count2)
    sum1 = sum1 + math.log(phi)
    sum2 = sum2 + math.log(1-phi)
    
    val = max(sum1,sum2)
    if val == sum1:
        res[i] = 1
    else:
        res[i] = 0
    if y[i] == res[i]:
        q = q + 1

print(res)

print(float(q)/float(m))
