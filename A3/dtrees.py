from xclib.data import data_utils
import scipy
import numpy as np
import math
# Read sparse file
labels = data_utils.read_sparse_file('train_x.txt')

x =scipy.sparse.csr_matrix.todense(labels)

f = open("train_y.txt","r")
contents = f.read()
y1 = list(map(int,contents.split()))

y = np.array(y1)


max = 0
index = 0
p1 = float(np.sum(y==1))/len(x)
e = -p1*math.log(p1)-(1-p1)*math.log(1-p1)
print(e)

for i in range(0,482):
    col = np.asarray(x[:,i]).reshape(-1)
    N = len(col)
    val = np.unique(col)
    entropy = e
    for j in range(0,len(val)):
        z = np.where(col == val[j])
        n = len(z[0])
        y2 = np.take(y,z[0])
        p = float(np.sum(y2 == 1))/n
        if p == 0 or p==1:
            h = 0
        else:
            h = -p*math.log(p)-math.log((1-p))*(1-p)
        entropy = entropy - n*h/float(N)
    if entropy > max:
        max = entropy
        index = i


class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.index = None
        self.median = None

root = Tree()
root.index = index
root.median = np.median(np.asarray(x[:,index]).reshape(-1))
#print(root,root.index,root.median,root.left,root.right)
# print(entropy)
print(index,max)
