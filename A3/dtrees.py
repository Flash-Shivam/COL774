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

class Node:
    def __init__(self):
        self.index = -1
        self.label = -1
        self.leaf = False
        self.median = -1
        self.split = True

    def set_leaf(self):
        self.leaf = True

    def set_index(self,index):
        self.index = index

    def set_label(self,label):
        self.label = label

    def set_median(self,median):
        self.median = median

    def add_child(self,num,node):
        if num == 0:
            self.left = node
        else:
            self.right = node

    def set_split(self,split):
        self.split = split


l = []
for i in range(0,len(x[0])):
    l.append(i)

columns = np.array(l)


def entropy(y):
    n = len(y)
    if n == 0:
        return 0
    p = np.sum(y==0)/n
    if p == 0 or p == 1:
        return 0
    else:
        return -p*math.log(p)-(1-p)*math.log(1-p)

def choose_best_attribute(x,y):

    e1 = entropy(y)
    # All y are same
    if e1 == 0:
        return y1[0],0,x,y,x,y,-1

    split = True
    max = -np.inf
    median = -1
    y_indicator = -1
    index1 = -1
    N = len(y)
    for i in range(0,len(x[0])):
        c1 = (x[:,i])
        #c1 = np.take(c,rows1)
        val = np.median(c1)
        entropy1 = e1
        f = np.amax(c1)
        if np.unique(c1).size == 1:
            entropy1 = -1
        else:
            if f == val:
                z = np.where(c1 < val)
            else:
                z = np.where(c1 <= val)
            y2 = np.take(y,z[0])
            n = len(y2)
            h0 = entropy(y2)
            entropy1 = entropy1 - n*h0/float(N)
            if f == val:
                z = np.where(c1 >= val)
            else:
                z = np.where(c1 > val)

            y3 = np.take(y,z[0])
            n = len(y3)
            h1 = entropy(y3)
            entropy1 = entropy1 - n*h1/float(N)

        if entropy1 > max :
            max = entropy1
            index1 = i
            median = val
            if f == val:
                split = False


    if max == -1:
        a,b = stats.mode(y1)
        return b[0],0,x,y,x,y,-1

    data_x = x[:, index1]

    if split:
        l_data = x[data_x <= median]
        r_data = x[data_x >  median]
    else:
        l_data = x[data_x <  median]
        r_data = x[data_x >= median]

    if split:
        l_label = y[data_x <= median]
        r_label = y[data_x >  median]
    else:
        l_label = y[data_x <  median]
        r_label = y[data_x >= median]

    a,b = stats.mode(y)
    return b[0],-1,l_data,l_label,r_data,r_label,index1


#root = Node(i1,v1,0)
c = 0
print("Grow Tree")

def grow_tree(x,y):
    global c
    c = c + 1

    label,indicator,l_d,l_l,r_d,r_l,index = choose_best_attribute(x,y)

    if indicator == 0:
        leaf = Node()
        leaf.set_leaf()
        leaf.set_label(label)
        return leaf

    left_node = grow_tree(l_d, l_l)
    right_node = grow_tree(r_d, r_l)

    median = np.median(x[:,index])

    node = Node()
    node.set_index(index)
    node.set_median(median)
    a,b = stats.mode(y)
    node.set_label(b[0])

    node.add_child(True,left_node)
    node.add_child(False,right_node)

    return node




start = time.process_time()

root = grow_tree(x,y)

print(time.process_time()-start)
print("Tree Built")

#print(i1,v1)

print(c)

def node_count(node):
    if node.leaf:
        return 1
    return 1 + node_count(node.left) + node_count(node.right)


# In[22]:


def node_height(node):
    if node.leaf:
        return 0
    return 1 + max(node_height(node.left), node_height(node.right))

print(node_count(root))

print(node_height(root))
