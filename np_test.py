import math
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

vectorizer = TfidfVectorizer(max_features=50)

# training.1600000.processed.noemoticon.csv
x = []
di0 = {}
filename = "training.1600000.processed.noemoticon.csv"
rows = []
tweets = []
y = []
cond = 0
clf_pf = GaussianNB()
batch = 1
count1 = 0
rows3= []
with open(filename, 'r', encoding='ISO-8859-1') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)
        rows3.append(row[5])
        if row[0] == "4":
            y.append(1)
        elif row[0] == "0":
            y.append(0)
        count1 = count1 + 1

#print(count1)
#print("Train Read OVer")
t = 0
filename1 = "testdata.manual.2009.06.14.csv"
rows1 = []

with open(filename1, 'r', encoding='ISO-8859-1') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if row[0] == "4" or row[0] == "0":
            t = t + 1
            rows1.append(row)
            rows3.append(row[5])

#print("Test Read OVer")
# print(t)
temp = t
y2 = np.zeros(t,dtype=float)
c = 0
x1 = []
for i in rows1:
    f_p = (i[5])
    x1.append(f_p)
    if i[0] == "4":
        y2[c] = 1.0
    elif i[0] == "0":
        y2[c] = 0.0
    c = c+1
    # print(p)
#print(len(y2))
#print("Y2 Filled")
X = vectorizer.fit_transform(rows3)
#print(X[3,:])
# print(type(X))
#print(X)
y1 = []
temp = []
#print("Sparse matrix")
cond = 0
for i in X:
    # print(list(i.A[0]))
    temp.append(list(i.A[0]))
    # print(temp)
    y1.append(y[cond])
    cond = cond + 1
    if cond % 1600 == 0:
        X1 = np.array(temp)
        Y1 = np.array(y1)
        # print(X1)
        #print(len(Y1),len(X1),cond)
        clf_pf.partial_fit(X1, Y1,[0,1])
        y1 = []
        temp = []
        batch = batch + 1
    if cond == count1:
        break

#print("trained")


#print(batch)
clf = GaussianNB()
#print("test")
temp = []
cond = 0
for i in X:
    cond = cond + 1
    if cond > count1:
        temp.append(list(i.A[0]))


X1 = np.array(temp)
#print(len(X1))

y3 = clf_pf.predict(X1)
#print("Calculating accuracy")
q = 0.0
for i in range(0,len(y2)):
    if y2[i] == y3[i]:
        q = q + 1

accuracy = float(q)/float(t)
print(accuracy)

#print(y3)
