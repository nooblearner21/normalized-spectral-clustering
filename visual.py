import numpy as np
import matplotlib.pyplot as plt


#test case of index input
clusters_points_indexes = [[1, 2], [7, 8, 3], [5, 6], [4, 0, 9]]
k = len(clusters_points_indexes)


#labeling each to each point its cluster index,change to n afterwards
def build_labels(n, clusters_array):
    #change to N
    labels = np.zeros(n, dtype=int)

    for cluster in range(len(clusters_array)):
        for vector_index in clusters_array[cluster]:
            labels[int(vector_index)] = cluster

    return labels


# two dimensions case
def visual_2d(observations, npc_labels, kmeans_labels):

    x = observations[0:, 0]
    y = observations[0:, 1]

    fig, axs = plt.subplots(1, 2, figsize=plt.figaspect(0.5))

    axs[0].scatter(x, y, c=npc_labels, cmap='rainbow')
    axs[0].set_title('NPC results')

    axs[1].scatter(x, y, c=kmeans_labels, cmap='rainbow')
    axs[1].set_title('KMeans results')

    fig.suptitle(f"Data was generated from the values: \n"
                 f"n = {len(observations)} , d = {len(observations[0])} \n"
                 f"The k that was used for both algorithms was {k} \n"
                 f"The Jaccard measure for Spectral Clustering: \n"
                 f"The Jaccard measure for K-means: ")

    plt.subplots_adjust(top=0.75)

    plt.savefig('figure.pdf')


# three dimensions case
def visual_3d(observations, npc_labels, kmeans_labels):

    x = observations[0:, 0]
    y = observations[0:, 1]
    z = observations[0:, 2]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax_2 = fig.add_subplot(121, projection='3d')
    ax_2.scatter(x, y, z, c=npc_labels, cmap="rainbow")
    ax_2.set_title('NPC results')

    bx_2 = fig.add_subplot(122, projection='3d')
    bx_2.scatter(x, y, z, c=kmeans_labels, cmap='rainbow')
    bx_2.set_title('KMeans results')

    fig.suptitle(f"Data was generated from the values: \n"
                 f"n = {len(observations)} , d = {len(observations[0])} \n"
                 f"The k that was used for both algorithms was {k} \n"
                 f"The Jaccard measure for Spectral Clustering: \n"
                 f"The Jaccard measure for K-means: ")

    plt.subplots_adjust(top=0.75)

    plt.savefig('figure.pdf')


def visual(observations, npc_labels, kmeans_labels):
    if (len(observations[0]) == 2):
        visual_2d(observations, npc_labels, kmeans_labels)
    elif (len(observations[0]) == 3):
        visual_3d(observations, npc_labels, kmeans_labels)


