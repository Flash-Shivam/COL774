from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import load_npz
import numpy as np
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

estimators = [50, 150, 250, 350, 450]
samples_split = [2, 4, 6, 8, 10]
features = [0.1, 0.3, 0.5, 0.7, 1.0]


a = 350
b = 10
c = 0.1

l1 = []
l2 = []
l3 = []


for i in features:
    clf = RandomForestClassifier(n_estimators=a, min_samples_split=b, max_features=i)
    clf.fit(x, y)
    y_2 = clf.predict(x_t)
    # print(y_2)
    y_3 = clf.predict(x_v)
    # print(y_3)
    acc1 = np.sum(y_3 == y3) / len(y3)
    acc2 = np.sum(y_2 == y2) / len(y2)
    l1.append(acc1)
    l2.append(acc2)
    l3.append(i)

print(l1, l2)

l4 = np.array(l3)
plt.xlim(l4.min() - 0.2, l4.max() + 0.2)

v_p = plt.plot(l3, l2, color='green', linestyle='dashed', linewidth=2, marker='o', markerfacecolor='green',
               markersize=2, label='validation')

t_p = plt.plot(l3, l1, color='red', linestyle='dashed', linewidth=2, marker='o', markerfacecolor='red',
               markersize=2, label='test')

plt.legend()

plt.xlabel('max features')

plt.ylabel('accuracy')

plt.show()


l1 = []
l2 = []
l3 = []

for i in estimators:
    clf = RandomForestClassifier(n_estimators=i, min_samples_split=b, max_features=c)
    clf.fit(x, y)
    y_2 = clf.predict(x_t)
    # print(y_2)
    y_3 = clf.predict(x_v)
    # print(y_3)
    acc1 = np.sum(y_3 == y3) / len(y3)
    acc2 = np.sum(y_2 == y2) / len(y2)
    l1.append(acc1)
    l2.append(acc2)
    l3.append(i)

print(l1, l2)

l4 = np.array(l3)
plt.xlim(l4.min() - 50, l4.max() + 50)

v_p = plt.plot(l3, l2, color='green', linestyle='dashed', linewidth=2, marker='o', markerfacecolor='green',
               markersize=2, label='validation')

t_p = plt.plot(l3, l1, color='red', linestyle='dashed', linewidth=2, marker='o', markerfacecolor='red',
               markersize=2, label='test')

plt.legend()

plt.xlabel('n_estimators')

plt.ylabel('accuracy')

plt.show()
