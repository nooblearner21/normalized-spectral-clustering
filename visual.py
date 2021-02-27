import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#redundent to grab the points themselves??
#clusters_points = []

#for i in range(k):
    #curr_cluster_points = [X_1[x] for x in clusters_points_indexes[i]]
    #clusters_points.append(curr_cluster_points)

#test case of index input
clusters_points_indexes = [[1, 2], [7, 8, 3], [5, 6], [4, 0, 9]]
k = len(clusters_points_indexes)

#labeling each to each point its cluster index,change to n afterwards
labels = np.zeros(shape=10)
print(labels)

for i in range(k):
    for vector_index in clusters_points_indexes[i]:
        labels[vector_index] = i


#visual output code, simple test case
fig = plt.figure()

#two dimensions case
X_1, labels_1 = make_blobs(n_samples=10, n_features=2)
x_1 = X_1[0:, 0]
y_1 = X_1[0:, 1]

ax_1 = fig.add_subplot(211)
ax_1.scatter(x_1, y_1, c=labels_1, cmap='RdBu')


#three dimensions case
X_2, labels_2 = make_blobs(n_samples=10, n_features=3)
x_2 = X_2[0:, 0]
y_2 = X_2[0:, 1]
z_2 = X_2[0:, 2]

ax_2 = fig.add_subplot(212, projection='3d')
ax_2.scatter(x_2, y_2, z_2, c=labels_2, cmap="RdBu")

fig.suptitle("Project by Dan and Yoni")

plt.show()




