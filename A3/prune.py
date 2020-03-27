from scipy import stats
import numpy as np
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
# Read sparse file
x = load_npz('train_x.npz').toarray()

x_t = load_npz('test_x.npz').toarray()

x_v = load_npz('valid_x.npz').toarray()


f = open("train_y.txt", "r")
contents = f.read()
y1 = list(map(int, contents.split()))

y = np.array(y1)

f = open("test_y.txt","r")
contents = f.read()
y1 = list(map(int, contents.split()))

y2 = np.array(y1)

f = open("valid_y.txt", "r")
contents = f.read()
y1 = list(map(int, contents.split()))

y3 = np.array(y1)


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
            return predict(node, sample)
        elif v < value:
            node = node.left
            return predict(node, sample)
        elif split:
            node = node.left
            return predict(node, sample)
        else:
            node = node.right
            return predict(node, sample)


def cal_accuracy(data, label, node):
    n = len(data)
    y4 = []
    for i in range(0,n):
        y_p = predict(node, data[i])
        y4.append(y_p)
    y5 = np.array(y4)
    return np.sum(y5 == label)/n


root = grow_tree(x, y, 50)

print("Tree Grown")


def bfs(node):
    node_list = [node]
    l_nodes = []
    while node_list:
        n = node_list.pop(0)
        if n.leaf:
            continue
        l_nodes.append(n)
        node_list.append(n.left)
        node_list.append(n.right)
    return l_nodes


def prune(node_list):
    b_score = cal_accuracy(x_v, y3, root)
    while node_list:
        n = node_list[-1]
        n.leaf = True
        score = cal_accuracy(x_v, y3, root)
        if score > b_score:
            return True
        n.leaf = False
        node_list.pop()
    return False


nodes = []
acc1 = []
acc2 = []
acc3 = []

while True:
    # print(len(bfs_l))
    bfs_l = bfs(root)
    if prune(bfs_l):
        nodes.append(node_count(root))
        a1 = cal_accuracy(x_v, y3, root)
        a2 = cal_accuracy(x_t, y2, root)
        a3 = cal_accuracy(x, y, root)
        acc1.append(a1)
        acc2.append(a2)
        acc3.append(a3)
        # print(a1, a2, a3)
        continue
    break

v_p = plt.plot(nodes, acc1, color='green', linestyle='dashed', linewidth=2, marker='o', markerfacecolor='green',
               markersize=2, label = 'validation')

t_p = plt.plot(nodes, acc2, color='red', linestyle='dashed', linewidth=2, marker='o', markerfacecolor='red',
               markersize=2, label = 'test')

tr_p = plt.plot(nodes, acc3, color='blue', linestyle='dashed', linewidth=2, marker='o', markerfacecolor='blue',
                markersize=2, label='train')
plt.xlabel('Number of Nodes')
# naming the y axis
plt.ylabel('Accuracy')

# giving a title to my graph
plt.title('D-tree with pruning')

# function to show the plot
plt.show()

