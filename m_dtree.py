from scipy import stats
import numpy as np
import time
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
PATH = '../data'


def read_data(name):
    feature = load_npz(PATH + name + '_x.npz').toarray()
    return feature


x = read_data('train')

x_t = read_data('test')

x_v = read_data('valid')


def read_label(name):
    f = open(PATH + name + "_y.txt", "r")
    contents = f.read()
    y1 = list(map(int, contents.split()))
    y_prime = np.array(y1)
    return y_prime


y = read_label('train')

y2 = read_label('test')

y3 = read_label('valid')


class Node:
    def __init__(self):
        self.index = -1
        self.label = -1
        self.leaf = False
        self.median = -1
        self.split = True

    def set_leaf(self):
        self.leaf = True

    def set_index(self, index):
        self.index = index

    def set_label(self, label):
        self.label = label

    def set_median(self, median):
        self.median = median

    def add_child(self, num, node):
        if num == 0:
            self.left = node
        else:
            self.right = node

    def set_split(self, split):
        self.split = split


feature_indexes = np.arange(len(x[0]))


def entropy(yP):
    n = len(yP)
    if n == 0:
        return 0
    p = np.sum(yP == 0)/n
    if p == 0 or p == 1:
        return 0
    else:
        return -p*np.log(p)-(1-p)*np.log(1-p)


def median_split(xP, yP, val):
    f = np.amax(xP)
    if f == val:
        r = np.where(xP < val)
        s = np.where(xP >= val)
        return yP[r[0]], yP[s[0]], False
    r1 = np.where(xP <= val)
    s1 = np.where(xP > val)
    return yP[r1[0]], yP[s1[0]], True


def mutual_information(xP, yP):
    val = np.median(xP)
    if np.unique(xP).size == 1:
        return -1, False

    left_y, right_y, sp = median_split(xP,yP,val)
    l_entropy = entropy(left_y)
    r_entropy = entropy(right_y)

    p = len(left_y)/len(yP)
    p1 = len(right_y)/len(yP)
    return entropy(yP) - (p*l_entropy + p1*r_entropy), sp


d = []

for i in range(0, len(x[0])):
    d.append(i)

columns = np.array(d)


def grow_tree(data,label,h):
    if entropy(label) == 0 or h == 0:
        leaf = Node()
        leaf.set_leaf()
        a, b = stats.mode(label)
        leaf.set_label(a[0])
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
        a, b = stats.mode(label)
        leaf.set_label(a[0])
        return leaf

    median = np.median(data[:, index])
    data_x = data[:, index]

    if split:
        x1 = np.where(data_x <= median)
        x1 = np.ix_(np.array(x1[0]), columns)
        l_data = data[x1]
        x2 = np.where(data_x > median)
        x2 = np.ix_(np.array(x2[0]), columns)
        r_data = data[x2]
    else:
        x1 = np.where(data_x < median)
        x1 = np.ix_(np.array(x1[0]), columns)
        l_data = data[x1]
        x2 = np.where(data_x >= median)
        x2 = np.ix_(np.array(x2[0]), columns)
        r_data = data[x2]
    if split:
        x1 = np.where(data_x <= median)
        l_label = label[x1[0]]
        x2 = np.where(data_x > median)
        r_label = label[x2[0]]
    else:
        x1 = np.where(data_x < median)
        l_label = label[x1[0]]
        x2 = np.where(data_x >= median)
        r_label = label[x2[0]]

    left_node = grow_tree(l_data, l_label, h-1)

    right_node = grow_tree(r_data, r_label, h-1)

    node = Node()
    node.set_index(index)
    node.set_median(median)
    node.set_split(split)
    a, b = stats.mode(y)
    node.set_label(a[0])

    node.add_child(0, left_node)
    node.add_child(1, right_node)

    return node


def node_count(node):
    if node.leaf:

        return 1
    return 1 + node_count(node.left) + node_count(node.right)


def node_height(node):
    if node.leaf:
        return 0
    return 1 + max(node_height(node.left), node_height(node.right))


def predict(node, sample):
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


def cal_accuracy(data, label, node):
    n = len(data)
    y4 = []
    for i in range(0,n):
        y_p = predict(node,data[i])
        y4.append(y_p)
    y5 = np.array(y4)
    return np.sum(y5 == label)/n


train_acc = []
test_acc = []
valid_acc = []
num_nodes = []
for i in range(1, 12, 1):
    start = time.process_time()
    root = grow_tree(x, y, i)
    print(time.process_time() - start)
    a1 = cal_accuracy(x, y, root)
    a2 = cal_accuracy(x_t, y2, root)
    a3 = cal_accuracy(x_v, y3, root)
    num_nodes.append(node_count(root))
    print(a1,a2,a3)
    train_acc.append(a1)
    test_acc.append(a2)
    valid_acc.append(a3)

for i in range(12, 30, 2):
    start = time.process_time()
    root = grow_tree(x, y, i)
    print(time.process_time() - start)
    a1 = cal_accuracy(x, y, root)
    a2 = cal_accuracy(x_t, y2, root)
    a3 = cal_accuracy(x_v, y3, root)
    num_nodes.append(node_count(root))
    print(a1,a2,a3)
    train_acc.append(a1)
    test_acc.append(a2)
    valid_acc.append(a3)

print(train_acc)
print(test_acc)
print(valid_acc)
print(num_nodes)

plt.plot(num_nodes, train_acc, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue',
         markersize=10)

plt.xlabel('Number of Nodes')
# naming the y axis
plt.ylabel('Accuracy')

# giving a title to my graph
plt.title('D-tree')

# function to show the plot
plt.show()

plt.plot(num_nodes, test_acc, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue',
         markersize=10)

plt.xlabel('Number of Nodes')
# naming the y axis
plt.ylabel('Accuracy')

# giving a title to my graph
plt.title('D-tree')

# function to show the plot
plt.show()

plt.plot(num_nodes, valid_acc, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue',
         markersize=10)

plt.xlabel('Number of Nodes')
# naming the y axis
plt.ylabel('Accuracy')

# giving a title to my graph
plt.title('D-tree')

# function to show the plot
plt.show()

