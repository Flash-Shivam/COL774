from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import load_npz
import numpy as np
import matplotlib.pyplot as plt

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

estimators = [50, 150, 250, 350, 450]
features = [0.1, 0.3, 0.5, 0.7, 1.0]
samples_split = [2, 4, 6, 8, 10]

a = 350
b = 10
c = 0.1

l1 = []
l2 = []
l3 = []

for i in samples_split:
    clf = RandomForestClassifier(n_estimators=i, min_samples_split=i, max_features=c)
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

plt.xlabel('min split')

plt.ylabel('accuracy')

plt.show()

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
plt.xlim(l4.min() - 2, l4.max() + 2)

v_p = plt.plot(l3, l2, color='green', linestyle='dashed', linewidth=2, marker='o', markerfacecolor='green',
               markersize=2, label='validation')

t_p = plt.plot(l3, l1, color='red', linestyle='dashed', linewidth=2, marker='o', markerfacecolor='red',
               markersize=2, label='test')

plt.legend()

plt.xlabel('max features')

plt.ylabel('accuracy')

plt.show()


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
