import csv
import math
import numpy as np

vocab = []
x = []
filename = "training.1600000.processed.noemoticon.csv"
rows = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)


print("Rows Appended")

for i in rows:
    p = (i[5]).split(" ")
    # print(p)
    sample = []
    for j in p:
        if j != "":
            vocab.append(j)
            sample.append(j)
    x.append(sample)

print("Vocab Formed")
print(type(vocab[0]))

dict1 = {}

for i in vocab:
    c = len(dict1) + 1
    dict1[i] = c

print(len(dict1))
v = len(dict1)
m = len(rows)
print("Dict Made")
y = np.zeros(m,dtype=int)
count = 0
for j in range(0,len(rows)):
    if rows[j][0] == '4':
        y[j] = 1
        count = count + 1

phi = float(count)/float(m)
print(phi,v)

l_phi = np.ones(v, dtype=float)
l_phi1 = np.ones(v, dtype=float)
count1 = v
count2 = v
for i in range(0,m):
    n = len(x[i])
    for j in range(0,n):
        k = dict1[x[i][j]] - 1
        if y[i] == 1:
            l_phi[k] = l_phi[k] + 1
        else:
            l_phi1[k] = l_phi1[k] + 1
    if y[i] == 1:
        count1 = count1 + n
    else:
        count2 = count2 + n
res1 = 1/float(count1)
res2 = 1/float(count2)
l_phi = np.dot(l_phi,res1)
l_phi1 = np.dot(l_phi1,res2)

res = np.zeros(m,dtype=float)
q = 0.0


for i in range(0,m):
    n = len(x[i])
    sum1 = 0.0
    sum2 = 0.0
    for j in range(0,n):
        sum1 = sum1 + math.log(l_phi[j])
        sum2 = sum2 + math.log(l_phi1[j])
    sum1 = float(phi*math.exp(sum1))
    sum2 = float((1-phi)*math.exp(sum2))
    if sum1 + sum2 == 0:
        sum1 = 0.1
    res[i] = float(sum1)/float(sum1+sum2)
    if res[i] >= 0.5:
        res[i] = 1
    if y[i] == res[i]:
        q = q + 1

print(float(q)/float(m))
