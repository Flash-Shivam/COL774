#!/usr/bin/env python
# coding: utf-8

# In[1]:


PATH = '/home/shivam/Videos/COL774(ML)/Ass3/data/'


# ### Imports

# In[2]:


from scipy.sparse import load_npz

import numpy as np
import pandas as pd

from collections import Counter

from typing import Dict, List, Tuple
from array import ArrayType

import matplotlib.pyplot as plt

import time


# In[3]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# ## Reading the data

# In[4]:


def read_data(name):
    _data = load_npz(PATH + name + '_x.npz').toarray()
    _labels = np.array(list(map(lambda x: int(x[0].strip()), open(PATH + name + '_y.txt').readlines())))
    # extracting unique values
    _unq = np.unique(_data, return_index=True, axis=0)[1]
    # filtering
    _data = _data[_unq]
    _labels = _labels[_unq]
    return _data, _labels


# In[5]:


train_d, train_l = read_data('train')
val_d, val_l = read_data('valid')
test_d, test_l = read_data('test')


# ### Plotting function

# In[6]:


def get_figure(title, x_label, y_label, x_lim, y_lim):
    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=12)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize='medium')
    if x_lim != None: ax.set_xlim(x_lim)
    if y_lim != None: ax.set_ylim(y_lim)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    return fig, ax


# ### DTrees Class

# In[7]:


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


# ## Learning Algorithm

# In[9]:


indices = np.arange(train_d.shape[1])
print(train_d.shape[1])
print(indices)
node_count = 0


# #### Entropy

# In[10]:


def entropy(labels):
    if(labels.size == 0): return 0
    p0 = labels[labels == 0].size / labels.size
    p1 = labels[labels == 1].size / labels.size
    if p0 == 0 or p1 == 0:
        return 0
    return - (p1 * np.log(p1) + p0 * np.log(p0))


# #### Median Splitting

# In[11]:


def median_split(array):
    max_ = array.max()
    min_ = array.min()
    median = np.median(array)
    if median == max_:
        if median == min_:
            raise ValueError('Complete array has only one single value. Error.')
        return ([array < median], [array >= median], False, median)
    return (array <= median, array > median, True, median)


# #### Mutual Information

# In[12]:


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
    global node_count
    node_count += 1

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
    for i in indices:
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


# ### Testing

# In[14]:


def test_value_with_tree(root, data_value):
    node = root
    while True:
        if node.leaf:
            return node.label
        ind = node.index
        spl = node.split

        if data_value[ind] < spl:
            node = node.left
        elif data_value[ind] > spl:
            node = node.right
        elif node.split_t:
            node = node.left
        else:
            node = node.right


# In[50]:


def test(values, label, root):
    _foo = np.vectorize(lambda x: test_value_with_tree(root, values[x]))
    result = _foo(np.arange(label.size))
    error = abs(label - result).sum() / label.size * 100
    return 100 - error


# ### Constructing various trees, in increasing order of height

# In[17]:


# Trying different tree sizes and testing on them
_node = []
_accuracy = []
_height = []

# height loop

for height in np.arange(2, 31, 2):

    print('Starting with height',height)

    global node_count1
    node_count1 = 0

    root = recursive_learn(train_d, train_l, height)

    acc_v = test(val_d, val_l, root)
    acc_tr = test(train_d, train_l, root)
    acc_ts = test(test_d, test_l, root)

    print(acc_v, acc_tr, acc_ts, height, node_count1)

    _node.append(node_count1)
    _accuracy.append([acc_tr, acc_v, acc_ts])
    _height.append(height)

print(_node)
# ### Plotting Height vs Accuracy

# In[23]:
'''

_accuracy = np.array(_accuracy)
fig, ax = get_figure('Plot', 'Tree Height', 'Accuracy', y_lim=[0, 100])
plt.plot(_height, _accuracy[:,0], 'rx--', label='train-data')
plt.plot(_height, _accuracy[:,1], 'bx--', label='val-data')
plt.plot(_height, _accuracy[:,2], 'gx--', label='test-data')
plt.legend()
fig.savefig('height_accuracy.png')


# ### Plotting Node-Count vs Accuracy

# In[24]:


fig, ax = get_figure('Plot', 'Number of nodes', 'Accuracy', y_lim=[0, 100])
plt.plot(_node, _accuracy[:,0], 'rx--', label='train-data')
plt.plot(_node, _accuracy[:,1], 'bx--', label='val-data')
plt.plot(_node, _accuracy[:,2], 'gx--', label='test-data')
plt.legend()
fig.savefig('node_accuracy.png')


# #### Tree related algorithms

# In[21]:


def node_count(node):
    if node.leaf:
        return 1
    return 1 + node_count(node.left) + node_count(node.right)


# In[22]:


def node_height(node):
    if node.leaf:
        return 0
    return 1 + max(node_height(node.left), node_height(node.right))


# In[31]:


def bfs(node):
    to_check = [node]
    bfs_list = []
    while to_check:
        _n = to_check.pop(0)
        if _n.leaf: continue
        bfs_list.append(_n)
        to_check.append(_n.left)
        to_check.append(_n.right)
    return bfs_list


# ### Pruning the tree obtained

# In[61]:


def best_prune(bfs_list):
    best_score = test(val_d, val_l, root)
    while bfs_list:
        node = bfs_list[-1]
        node.leaf = True
        score = test(val_d, val_l, root)
        if score > best_score:
            return True
        node.leaf = False
        bfs_list.pop()
    return False


# In[56]:


_node = []
_accuracy = []


# In[115]:


while True:
    bfs_l = bfs(root)
    if best_prune(bfs_l):
        _node.append(node_count(root))
        acc_v = test(val_d, val_l, root)
        acc_r = test(train_d, train_l, root)
        acc_s = test(test_d, test_l, root)
        _accuracy.append([acc_r, acc_v, acc_s])
        continue
    break


# In[116]:


_accuracy, _node


# ## Plotting pruning vs. accuracies

# In[118]:


x = np.array(_accuracy)
fig, ax = get_figure('Plot', 'Node count (as pruned)', 'Accuracy', y_lim=[75, 100])
plt.plot(_node, x[:,0], 'rx--', lw=1, label='train-data')
plt.plot(_node, x[:,1], 'bx--', lw=1, label='val-data')
plt.plot(_node, x[:,2], 'gx--', lw=1, label='test-data')
plt.gca().invert_xaxis()
plt.legend()
fig.savefig('qb.png')
plt.show()
'''
