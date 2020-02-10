import numpy as np
import csv
filename = "q2test.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

y = np.zeros(10000,dtype=float)
z = np.zeros((10000,3),dtype=float)
for i in range(0,10000):
    y[i] = rows[i][2]
    z[i, 0] = 1
    z[i, 1] = rows[i][0]
    z[i, 2] = rows[i][1]

theta = np.zeros(3,dtype=float)

theta[0] = 3.00371
theta[1] = 0.99475
theta[2] = 1.9981


def error(theta,y,z,m):
    p = 0
    for i in range(0,m):
        p = p + ((y[i]-(theta[1]*z[i, 1]+theta[2]*z[i, 2]+theta[0]))**2)/(2*m)
    return p


print(error(theta,y,z,10000))

theta[0] = 2.922
theta[1] = 1.01
theta[2] = 1.99

print(error(theta,y,z,10000))
