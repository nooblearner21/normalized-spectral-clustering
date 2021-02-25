import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#need to grab input instead

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




