import csv
import numpy as np
filename = "train.csv"
rows = []
rows1 = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row[:-1])
        rows1.append(int(row[-1]))


def hot_encode(length, elements):
    y = np.zeros((length, 26), dtype=int)
    for i in range(0, length):
        y[i][elements[i]] = 1
    return y


def sigmoid(x):
    return 1/np.exp(-x)


def ds(x):
    return x*(1-x)


x = np.array(rows, dtype=int)
y = hot_encode(len(x), rows1)

r = len(y[0])
m = len(x)
n = len(x[0])
M = 100
hidden_layer = [100, 50]

h = [n]
for i in hidden_layer:
    h.append(i)
h.append(r)


def find_dim(layers):
    w1 = []
    b1 = []
    for i in range(0,len(layers)-1):
        w1.append((layers[i+1], layers[i]))
        b1.append(layers[i+1])
    return w1, b1


w_list, b_list = find_dim(h)
w = list()
b = list()


def initialize_weight(w_l, b_l, layer):
    global w
    global b
    for i in range(0, len(w_l)):
        c1,c2 = w_l[i]
        b.append(np.zeros((b_l[i], 1), dtype=float))
        var = np.sqrt(2/(layer[i+1]+layer[i]))
        v = np.array(np.random.rand(c1, c2), dtype=float)
        for j in range(0, c1):
            mu = np.mean(v[j])
            sigma = np.var(v[j])
            v[j] = (v[j]-mu)/(sigma*var)
        w.append(v)
    return


initialize_weight(w_list, b_list, h)
