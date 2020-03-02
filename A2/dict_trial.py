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

for i in rows:
    p = (i[5]).split(" ")
    # print(p)
    sample = []
    for j in p:
        if j != "":
            vocab.append(j)
            sample.append(j)
    x.append(sample)

print(type(vocab[0]))

unique_words = []
print(len(vocab))

unique_words = list(set(vocab))
unique_words.sort()
print(len(unique_words))
v = len(unique_words)
dict1 = {}
count = 1
for i in unique_words:
    dict1[i] = count
    count = count + 1
m = len(rows)
y = np.zeros(m,dtype=int)
count = 0
for j in range(0,len(rows)):
    if rows[j][0] == '4':
        y[j] = 1
        count = count + 1

phi = float(count)/float(m)
l_phi = np.zeros(v, dtype=float)
l_phi1 = np.zeros(v, dtype=float)
for k in range(0,v):
    count2 = 1
    count3 = 1
    count4 = v
    count5 = v
    for i in range(0,m):
        n = len(x[i])
        if y[i] == 1:
            for j in range(0,n):
                if dict1[x[i][j]] == k:
                    count2 = count2 + 1
            count4 = count4 + n
        else:
            for j in range(0,n):
                if dict1[x[i][j]] == k:
                    count3 = count3 + 1
            count5 = count5 + n
    l_phi[k] = float(count2)/float(count4)
    l_phi1[k] = float(count3)/float(count5)


res = np.zeros(m,dtype=float)
q = 0
for i in range(0,m):
    n = len(x[i])
    sum1 = 0
    sum2 = 0
    for j in range(0,n):
        sum1 = sum1 + math.log(l_phi[j])
        sum2 = sum2 + math.log(l_phi1[j])
    sum1 = phi*math.exp(sum1)
    sum2 = (1-phi)*math.exp(sum2)
    res[i] = float(sum1)/float(sum1+sum2)
    if res[i] >= 0.5:
        res[i] = 1
    if y[i] == res[i]:
        q = q + 1

print(float(q)/float(m))
