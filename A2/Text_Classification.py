import csv
import numpy as np
import math
filename = "training.1600000.processed.noemoticon.csv"
rows = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)
vocab = []
for i in rows:
    p = (i[5].split(" "))
    for j in p:
        vocab.append(j)
print(len(vocab))
words = np.array(vocab)

unique_words = np.unique(words)

n = len(unique_words)
print(len(words))
m = len(rows)
print(n, m)
x = np.zeros((m, n),dtype= int)

y = np.zeros(m,dtype=int)
count = 0
for j in range(0,len(rows)):
    if rows[j][0] == '4':
        y[j] = 1
        count = count + 1


for i in rows:
    p = (i[5].split(" "))
    for j in p:
        res = np.where(unique_words == j)
        x[i, res[0][0]] = 1

phi = float(count)/float(m)

l_phi = np.zeros(n,dtype=float)

for i in range(0,n):
    r = 0
    for j in range(0,m):
        if y[j] == 1 and x[j, i] == 1:
            r = r + 1
    # Laplace Smoothing
    l_phi[i] = float(r+1)/float(count+2)

L_phi = np.zeros(n,dtype=float)

for i in range(0,n):
    L_phi[i] = math.log(l_phi[i])


