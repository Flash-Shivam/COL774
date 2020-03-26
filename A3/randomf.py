from xclib.data import data_utils
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
import scipy
import numpy as np
import math
import time
from sklearn.ensemble import RandomForestClassifier

labels = data_utils.read_sparse_file('train_x.txt')

x1 =scipy.sparse.csr_matrix.todense(labels)

x = np.asarray(x1)

f = open("train_y.txt","r")
contents = f.read()
y1 = list(map(int,contents.split()))

y = np.array(y1)

param_grid = [{'n_estimators' : [50 , 150, 250, 350, 450], 'min_samples_split' : [0.1, 0.3, 0.5, 0.7 ,1.0]  ,'max_features' : [2, 4, 6 , 8 , 10]}]

search = GridSearchCV(RandomForestClassifier(oob_score=True,n_jobs=-1), param_grid, cv=3)

search.fit(x, y)

print(sorted(search.cv_results_.keys()))
print("[]")

z1 = search.cv_results_['rank_test_score']

q = -1
for i in range(0,len(z1)):
    if z1[i] == 1:
        q = i
        break

#p,z = np.where(np.array(search.cv_results_['rank_test_score'])==1)
print(search.cv_results_['params'][q])
#print(search.cv_results_)
