import numpy as np
from tkinter import Tk, simpledialog, filedialog
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

matplotlib.rc('figure', figsize=(10, 5))
Tk().withdraw()
FILE_PATH = filedialog.askopenfilename()
fileValues = pd.read_csv(FILE_PATH, delimiter='\s+', header=None, names=["points", "y"])

DIM = 2
points = fileValues.iloc[:, [0, 1]].values
N = len(points)
y = np.zeros(N)
num_of_clusters = simpledialog.askinteger("Input", "Write amount of clusters!")
sse = []


def k_avg(num_cluster, x, y, N):
    flag = True
    is_first_iteration = True
    prev_centroid = []
    centroid = None
    avg_arr = []
    avg = 0

    while flag:
        if is_first_iteration:
            is_first_iteration = False
            start_point = np.random.choice(range(N), num_cluster, replace=False)
            centroid = x[start_point]
        else:
            prev_centroid = np.copy(centroid)
            for i in range(num_cluster):
                centroid[i] = np.mean(x[y == i], axis=0)
        for i in range(N):
            dist = np.sum((centroid - x[i]) ** 2, axis=1)
            avg_arr.append(min(dist))
            min_ind = np.argmin(dist)
            y[i] = min_ind
        if np.array_equiv(centroid, prev_centroid):
            avg = np.mean(avg_arr)
            flag = False
    avg_arr.clear()
    return avg, y, x


avg, y, x = k_avg(num_of_clusters, points, y, N)
sse.append(avg)
for k in range(num_of_clusters):
    fig = plt.scatter(x[y == k, 0], x[y == k, 1])
plt.show()

for i in range(1, num_of_clusters):
    y = np.copy(y)
    avg, y, x = k_avg(i, x, y, N)
    sse.append(avg)

sse.sort(reverse=True)
plt.plot(list(range(1, num_of_clusters + 1)), sse)
plt.xticks(list(range(1, num_of_clusters + 1)))
plt.scatter(list(range(1, num_of_clusters + 1)), sse)
plt.xlabel("Amount of Clusters")
plt.ylabel("SSE")
plt.show()
