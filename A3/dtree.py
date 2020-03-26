from xclib.data import data_utils
from scipy import stats
import scipy
import numpy as np
import math
import time
# Read sparse file
labels = data_utils.read_sparse_file('train_x.txt')

x =scipy.sparse.csr_matrix.todense(labels)

x = np.asarray(x)
#print(type(x))
f = open("train_y.txt","r")
contents = f.read()
y1 = list(map(int,contents.split()))

y = np.array(y1)

class Node(object):
    def __init__(self,index,value,indicator):
        self.left = None
        self.right = None
        self.index = index
        self.value = value
        self.indicator = indicator
        # self.median = None

l = []
for i in range(0,len(x[0])):
    l.append(i)

columns = np.array(l)

def choose_best_attribute(index,value,x,y,seg):
    max = -1
    median = 0
    y_indicator = -1
    index1 = 0

    length = len(x)

    w = []

    for i in range(0,length):
        w.append(i)
    w1 = np.array(w)
    rows1 = np.array(w)
    if index >= 0:
        col2 = (x[:,index])
        if seg == 0:
            z = np.where(col2 <= value)
        else:
            z = np.where(col2 > value)

        rows1 = np.intersect1d(w1,z[0])
        y1 = np.take(y,rows1)
    else:
        y1 = np.take(y,w1)

    count = 0
    N = len(rows1)

    p1 = float(np.sum(y1==1))/N
    if p1 == 0 or p1 == 1:
        e1 = 0
    else:
        e1 = -p1*math.log(p1)-(1-p1)*math.log(1-p1)
    for i in range(0,len(x[0])):
        c = (x[:,i])
        c1 = np.take(c,rows1)
        val = np.median(c1)
        entropy = 0.0

        z = np.where(c1 <= val)
        n = len(z[0])
        y2 = np.take(y1,z[0])
        if n != 0:
            p = float(np.sum(y2 == 1))/n
        else:
            p = 0
        if p == 0 or p==1:
            h = 0
        else:
            h = -p*math.log(p)-math.log((1-p))*(1-p)
        entropy = entropy - n*h/float(N)

        z = np.where(c1 > val)
        n = len(z[0])
        y2 = np.take(y1,z[0])
        if n != 0:
            p = float(np.sum(y2 == 1))/n
        else:
            p = 0
        if p == 0 or p==1:
            h = 0
        else:
            h = -p*math.log(p)-math.log((1-p))*(1-p)
        entropy = entropy - n*h/float(N)
        if entropy == 0.0:
            count = count + 1
        if entropy > max and entropy!= 0.0:
            max = entropy
            index1 = i
            median = np.median(c1)

            y3 = np.take(y,rows1)
            z1 = len(y3)
            f1 = np.sum(y3 == 0)
            if (np.unique(c1)).size == 1 or e1 + entropy == 0.0:
                y_indicator = -2
            elif f1 == 0 or f1 == z1:
                y_indicator = (z1 - f1)/z1
            else:
                y_indicator = -1

    x1 = x[np.ix_(rows1,columns)]

    if count == 482:
        return (-1,-1,-3,x1,y1)
    return (index1,median,y_indicator,x1,y1)

i1,v1,d1,d2,d3 = choose_best_attribute(-1,-1,x,y,-1)

root = Node(i1,v1,0)


def grow_tree(index,value,x,y,node):
    x1,y1,z1,s1,t1 = choose_best_attribute(index,value,x,y,0)
    x2,y2,z2,s2,t2 = choose_best_attribute(index,value,x,y,1)
    if z1 == -3:
        node.indicator = -1
        a,b = stats.mode(y1)
        node.left = Node(-1,b[0],0)
    elif z1 == -1:
        node.left = Node(x1,y1,0)
        grow_tree(x1,y1,s1,t1,node.left)
    elif z1 == -2:
        node.left = Node(x1,y1,-1)
        a,b = stats.mode(y1)
        node = node.left
        node.left = Node(-1,b[0],0)
    elif z1 >= 0:
        node.left = Node(x1,y1,-1)
        num1 = num1 + 1
        node = node.left
        node.left = Node(-1,z1,0)


    if z2 == -3:
        node.indicator = -1
        a,b = stats.mode(y2)
        node.left = Node(-1,b[0],0)

    elif z2 == -1:
        node.right = Node(x2,y2,0)
        grow_tree(x2,y2,s2,t2,node.right)
    elif z2 == -2:
        node.right = Node(x2,y2,-1)
        a,b = stats.mode(y2)
        node = node.right
        node.left = Node(-1,b[0],0)
    elif z2 >= 0:
        node.right = Node(x2,y2,-1)
        node = node.right
        node.left = Node(-1,z2,0)

start = time.process_time()

grow_tree(i1,v1,x,y,root)

print(time.process_time()-start)
print("Tree Built")

def get_predition(node,sample):
    if node.indicator == -1:
        h = node.left.value
        return int(h)
    else:
        median = node.value
        feature = node.index
        if sample[feature] <= median:
            return get_predition(node.left,sample)
        else:
            return get_predition(node.right,sample)


y_pred = []

for i in range(0,len(x)):
    t = get_predition(root,x[i])
    y_pred.append(int(t))

y4 = np.array(y_pred)

accuracy = np.sum(y4 == y)/(len(y))

print(accuracy)
