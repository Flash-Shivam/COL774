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

class Node:


    def __init__(self):
        self.index   = -1
        self.label   = -1
        self.leaf  = False
        self.range = None
        self.split_t = True

    def set_split_type(self, split_t):
        self.split_t = split_t

    def set_as_leaf(self):
        self.leaf = True

    def set_index(self, index):
        self.index = index

    def set_split(self, median):
        self.split = median

    def add_child(self, _node, left):
        assert(not self.leaf)
        if left:
            self.left = _node
        else:
            self.right = _node

    def set_label(self, label):
        self.label = label

def median_split(array):
    max_ = array.max()
    min_ = array.min()
    median = np.median(array)
    if median == max_:
        if median == min_:
            raise ValueError('Complete array has only one single value. Error.')
        return ([array < median], [array >= median], False, median)
    return (array <= median, array > median, True, median)

feature_indexes = np.arange(len(x[0]))

def entropy(labels):
    if(labels.size == 0): return 0
    p0 = labels[labels == 0].size / labels.size
    p1 = labels[labels == 1].size / labels.size
    if p0 == 0 or p1 == 0:
        return 0
    return - (p1 * np.log(p1) + p0 * np.log(p0))



def mutual_information(data, label):

    if(np.unique(data).size == 1):
        return -1, -1

    l_split, h_split, strict, median = median_split(data)

    H0 = entropy(label[l_split])
    H1 = entropy(label[h_split])

    p0 = label[l_split].size / label.size
    p1 = 1 - p0

    return entropy(label) - (p0 * H0 + p1 * H1), strict


# ### Algorithm

# In[13]:


def recursive_learn(data, label, height):

    # Keeping the node count

    # Checking the entropy
    ent = entropy(label)

    # 0 entropy, data is completely sorted, node must be leaf
    if ent == 0 or height == 0:
        leaf = Node()
        leaf.set_as_leaf()
        leaf.set_label(label[0])
        return leaf

    # Getting mutual information w.r.t. all attributes
    index = -1
    split = True
    l = []
    max_info = -np.inf
    for i in feature_indexes:
        info, split_t = mutual_information(data[:, i], label)
        l.append(info)
        if info >= max_info:
            index = i
            max_info = info
            split = split_t

    # Information is still -1, implies contradicting labels, all values must be the same
    # Assign as a leaf, keeping majority choice
    if max_info == -1:
        print('Fucking shit man.')
        leaf = Node()
        leaf.set_as_leaf()
        leaf.set_label(Counter(label).most_common(1)[0][0])
        return leaf

    # Getting the median on that index
    median = np.median(data[:, index])

    # Data column, working atrribute
    data_x = data[:, index]

    # Splitting data w.r.t. the new index
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

    # Left node recursion
    left_node = recursive_learn(l_data, l_label, height-1)

    # Right node recursion
    right_node = recursive_learn(r_data, r_label, height-1)

    # Creating the correct node
    node = Node()
    node.set_index(index)
    node.set_split(median)
    node.set_split_type(split)
    node.set_label(Counter(label).most_common(1)[0][0])

    node.add_child(left_node, True)
    node.add_child(right_node, False)

    return node

start = time.process_time()
root = recursive_learn(x, y, 30)
print(time.process_time()-start)

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
        if node.leaf:
            return node.label
        ind = node.index
        spl = node.median

        if data_value[ind] < spl:
            node = node.left
        elif data_value[ind] > spl:
            node = node.right
        elif node.split:
            node = node.left
        else:
            node = node.right


y1 = []
for i in range(0,len(x)):
    t = test_value_with_tree(root,x[i])
    y1.append(t)

y2 = np.array(y1)
print(y2,np.sum(y2==0),np.sum(y2==1),len(y2))
accuracy = (np.sum(y2 == y5))/len(y)

print(accuracy)
