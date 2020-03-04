import csv
import math
import numpy as np

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
pos = 0
neg = 0
x1 = []
for i in rows1:
    p = (i[5]).split()
    x1.append(p)
    if i[0] == "4":
        y1[c] = 1.0
        pos = pos + 1
    elif i[0] == "0":
        y1[c] = 0.0
        neg = neg + 1
    c = c+1



if pos > neg:
    y2 = np.ones(t,dtype=float)
else:
    y2 = np.zeros(t,dtype=float)

np.random.seed(0)
random_labels = np.random.randint(2,size=t)
random_accuracy = (y1 == random_labels).sum()/float(t)
print("Random accuracy = {0}".format(random_accuracy))

max_pred_accuracy = (y1 == y2).sum()/float(t)
print("Maximum Majority accuracy = {0}".format(max_pred_accuracy))
