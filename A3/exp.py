from xclib.data import data_utils
from scipy import stats
import scipy
import numpy as np
import math
import time
# Read sparse file
labels = data_utils.read_sparse_file('train_x.txt')

x1 =scipy.sparse.csr_matrix.todense(labels)

x = np.asarray(x1)

#x = x2[u]

f = open("train_y.txt","r")
contents = f.read()
y1 = list(map(int,contents.split()))

y = np.array(y1)
y5 = np.array(y1)
#y = y2[u]

#print(y_prime)
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


feature_indexes = np.arange(len(x[0]))

def entropy(yP):
    n = len(yP)
    if n == 0:
        return 0
    p = np.sum(yP==0)/n
    #print(p,n)
    if p == 0 or p == 1:
        return 0
    else:
        return -p*math.log(p)-(1-p)*math.log(1-p)


def median_split(xP,yP,val):
    f = np.amax(xP)
    if f == val:
        return yP[xP<val],yP[xP>=val],False
    return yP[xP<=val],yP[xP>val],True


def mutual_information(xP,yP):
    val = np.median(xP)
    if np.unique(xP).size == 1:
        return -1, False

    left_y, right_y, sp = median_split(xP,yP,val)
    l_entropy = entropy(left_y)
    r_entropy = entropy(right_y)

    p = len(left_y)/len(yP)
    p1 = len(right_y)/len(yP)
    return entropy(y) - (p*l_entropy + p1*r_entropy), sp


c = 0

d = []
e = 0
k = 0
for i in range(0,len(x[0])):
    d.append(i)

columns = np.array(d)
print(y)
def grow_tree(data,label,h):
    global c
    global e
    global k
    c = c + 1
    if entropy(label) == 0 or h == 0:
        leaf = Node()
        leaf.set_leaf()
        a,b = stats.mode(label)
        #print(a,b)
        #print(len(label),"66666",label[0],a[0])
        leaf.set_label(a[0])
        if a[0] == 0:
            e = e + 1
        elif a[0] == 1:
            k = k +1
        return leaf

    index = -1
    split = True
    max_info = -np.inf
    for i in feature_indexes:
        info, split_t = mutual_information(data[:, i], label)
        if info >= max_info:
            index = i
            max_info = info
            split = split_t

    if max_info == -1:
        leaf = Node()
        leaf.set_leaf()
        a,b = stats.mode(label)
        #print(a,b)
        #print(len(label),"####")
        leaf.set_label(a[0])
        if a[0] == 0:
            e = e + 1
        elif a[0] == 1:
            k = k +1
        return leaf

    median = np.median(data[:, index])
    data_x = data[:, index]
    '''
    if split:
        x1 = np.where(data_x <= median)
        x1 = np.ix_(np.array(x1[0]),columns)
        l_data = x[x1]
        x2 = np.where(data_x > median)
        x2 = np.ix_(np.array(x2[0]),columns)
        r_data = x[x2]
    else:
        x1 = np.where(data_x < median)
        x1 = np.ix_(np.array(x1[0]),columns)
        l_data = x[x1]
        x2 = np.where(data_x >= median)
        x2 = np.ix_(np.array(x2[0]),columns)
        r_data = x[x2]

    if split:
        x1 = np.where(data_x <= median)
        l_label = y[x1[0]]
        x2 = np.where(data_x > median)
        r_label = y[x2[0]]
    else:
        x1 = np.where(data_x < median)
        l_label = y[x1[0]]
        x2 = np.where(data_x >= median)
        r_label = y[x2[0]]
    '''
    if split:
        l_data = data[data_x <= median]
        r_data = data[data_x >  median]
    else:
        l_data = data[data_x <  median]
        r_data = data[data_x >= median]
    # Splitting label w.r.t. the new index
    if split:
        l_label = label[data_x <= median]
        r_label = label[data_x >  median]
    else:
        l_label = label[data_x <  median]
        r_label = label[data_x >= median]

    #print(l_label,r_label)
    left_node = grow_tree(l_data, l_label,h-1)

    right_node = grow_tree(r_data, r_label,h-1)


    node = Node()
    node.set_index(index)
    node.set_median(median)
    node.set_split(split)
    a,b = stats.mode(y)
    node.set_label(b[0])

    node.add_child(True,left_node)
    node.add_child(False,right_node)

    return node


start = time.process_time()
root = grow_tree(x,y,50)
print(time.process_time()-start)
print(c)
def node_count(node):
    if node.leaf:
        #print(node.label)
        return 1
    return 1 + node_count(node.left) + node_count(node.right)

def node_height(node):
    if node.leaf:
        return 0
    return 1 + max(node_height(node.left), node_height(node.right))

print(node_count(root))

print(node_height(root))

def predict(node,sample):
    if node.leaf:
        return node.label
    else:
        index = node.index
        value = node.median
        split = node.split
        v = sample[index]

        if v > value:
            node = node.right
            return predict(node,sample)
        elif v < value:
            node = node.left
            return predict(node,sample)
        elif split:
            node = node.left
            return predict(node,sample)
        else:
            node = node.right
            return predict(node,sample)

def test_value_with_tree(root, data_value):
    node = root
    while True:
        #print(node.index)

        if node.leaf:
            print("-------",node.label)
            return node.label
        ind = node.index
        spl = node.median
        print(ind,spl,data_value[ind],node.split)

        if data_value[ind] < spl:
            node = node.left
            print("L")
        elif data_value[ind] > spl:
            node = node.right
            print("R")
        elif node.split:
            node = node.left
            print("L")
        else:
            node = node.right
            print("R")

#print(root.left,root.left.left)
print(e,k)
y1 = []
for i in range(0,10):
    t = test_value_with_tree(root,x[i])
    print(";;;;;;;;",y[i])
    y1.append(t)

y2 = np.array(y1)
print(y2,np.sum(y2==0),np.sum(y2==1),len(y2))
accuracy = (np.sum(y2 == y5))/len(y)

print(accuracy)
