import numpy as np
import matplotlib.pyplot as plt


#labeling each to each point its cluster index,change to n afterwards
def build_labels(n, clusters_array):
    #change to N
    labels = np.zeros(n, dtype=int)

    for cluster in range(len(clusters_array)):
        for vector_index in clusters_array[cluster]:
            labels[int(vector_index)] = cluster

    return labels


# two dimensions case
def visual_2d(observations, npc_labels, kmeans_labels, spectral_measure, kmeans_measure, k, k_for_blobs):

    x = observations[0:, 0]
    y = observations[0:, 1]

    fig, axs = plt.subplots(1, 2, figsize=plt.figaspect(0.5))

    axs[0].scatter(x, y, c=npc_labels, cmap='rainbow')
    axs[0].set_title('NPC results')

    axs[1].scatter(x, y, c=kmeans_labels, cmap='rainbow')
    axs[1].set_title('KMeans results')

    fig.suptitle(f"Data was generated from the values: \n"
                 f"n = {len(observations)} , k = {k_for_blobs} \n"
                 f"The k that was used for both algorithms was {k} \n"
                 f"The Jaccard measure for Spectral Clustering: {spectral_measure}\n"
                 f"The Jaccard measure for K-means: {kmeans_measure}")

    plt.subplots_adjust(top=0.75)

    plt.savefig('figure.pdf')


# three dimensions case
def visual_3d(observations, npc_labels, kmeans_labels, spectral_measure, kmeans_measure, k, k_for_blobs):

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
                 f"n = {len(observations)} , k = {k_for_blobs} \n"
                 f"The k that was used for both algorithms was {k} \n"
                 f"The Jaccard measure for Spectral Clustering: {spectral_measure}\n"
                 f"The Jaccard measure for K-means: {kmeans_measure}")

    plt.subplots_adjust(top=0.75)

    plt.savefig('figure.pdf')


def visual(observations, npc_labels, kmeans_labels, spectral_measure, kmeans_measure, k, k_for_blobs):
    if (len(observations[0]) == 2):
        visual_2d(observations, npc_labels, kmeans_labels, spectral_measure, kmeans_measure, k, k_for_blobs)
    elif (len(observations[0]) == 3):
        visual_3d(observations, npc_labels, kmeans_labels, spectral_measure, kmeans_measure, k, k_for_blobs)


def jaccard_measure(blobs_labels, spectral_labels, kmeans_labels):
    n = len(blobs_labels)
    blobs_labels = blobs_labels.copy()
    spectral_labels = spectral_labels.copy()
    kmeans_labels = kmeans_labels.copy()

    jaccard_matrix_blobs = np.equal(blobs_labels.reshape(n, 1), blobs_labels.reshape(1, n))
    jaccard_matrix_spectral = np.equal(spectral_labels.reshape(n, 1), spectral_labels.reshape(1, n))
    jaccard_matrix_kmeans = np.equal(kmeans_labels.reshape(n, 1), kmeans_labels.reshape(1, n))

    spectral_intersection = np.logical_and(jaccard_matrix_blobs, jaccard_matrix_spectral)
    spectral_union = np.logical_or(jaccard_matrix_blobs, jaccard_matrix_spectral)

    spectral_measure = spectral_intersection.sum() / float(spectral_union.sum())

    kmeans_intersection = np.logical_and(jaccard_matrix_blobs, jaccard_matrix_kmeans)
    kmeans_union = np.logical_or(jaccard_matrix_blobs, jaccard_matrix_kmeans)

    kmeans_measure = kmeans_intersection.sum() / float(kmeans_union.sum())

    return spectral_measure, kmeans_measure









