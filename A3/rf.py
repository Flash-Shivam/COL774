from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import load_npz
import numpy as np
from sklearn.model_selection import GridSearchCV

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
