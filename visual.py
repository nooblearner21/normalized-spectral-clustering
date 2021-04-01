import numpy as np
import matplotlib.pyplot as plt

#test case of index input
clusters_points_indexes = [[1, 2], [7, 8, 3], [5, 6], [4, 0, 9]]
k = len(clusters_points_indexes)


#labeling each to each point its cluster index,change to n afterwards
def build_labels(n, clusters_array):
    #change to N
    labels = np.zeros(n)

    for cluster in range(len(clusters_array)):
        for vector_index in clusters_array[cluster]:
            labels[vector_index] = cluster


#two dimensions case
def visual_2d(observations, labels):
    fig = plt.figure()

    #X_1, labels_1 = make_blobs(n_samples=10, n_features=2)
    x = observations[0:, 0]
    y = observations[0:, 1]

    ax_1 = fig.add_subplot(211)
    ax_1.scatter(x, y, c=labels, cmap='RdBu')

    fig.suptitle("Project by Dan and Yoni")

    plt.show()

#three dimensions case
def visual_3d(observations, labels):
    fig = plt.figure()

    #X_2, labels_2 = make_blobs(n_samples=10, n_features=3)
    x = observations[0:, 0]
    y = observations[0:, 1]
    z = observations[0:, 2]

    ax_2 = fig.add_subplot(212, projection='3d')
    ax_2.scatter(x, y, z, c=labels, cmap="RdBu")

    fig.suptitle("Project by Dan and Yoni")

    plt.show()




