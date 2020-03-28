from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import load_npz
import numpy as np
from sklearn.model_selection import GridSearchCV
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

clf = RandomForestClassifier(oob_score=True)

param_grid = [{'n_estimators': [350], 'min_samples_split': [10], 'max_features': [0.1]}]

search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy',verbose=1,refit=True)

clf1 = search.fit(x, y)

print(sorted(clf1.cv_results_))

y_1 = clf1.predict(x)
print(y_1)

y_2 = clf1.predict(x_t)
print(y_2)
y_3 = clf1.predict(x_v)
print(y_3)
acc1 = np.sum(y_3 == y3)/len(y3)
acc2 = np.sum(y_2 == y2)/len(y2)
acc3 = np.sum(y_1 == y)/len(y)

print(acc1, acc2, acc3)
