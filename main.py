import argparse
import numpy as np
from sklearn.datasets import make_blobs

import visual
from ops import qr_decomposition, tmatrix, normalized_laplacian

import kmeanspp
from kmeans_pp import k_means_pp

parser = argparse.ArgumentParser()
parser.add_argument("k", type=int)
parser.add_argument("n", type=int)
parser.add_argument('--random', dest='random', action='store_true')
parser.add_argument('--no-random', dest='random', action='store_false')

args = parser.parse_args()

k = args.k
n = args.n
random = args.random


"""
Returns the corresponding cluster index ~ observation relationship
"""
def cluster(matrix_t, centroids):
    # create list of K lists
    index_list = [[] for i in range(len(centroids))]

    curr_row = -1
    for row in matrix_t:
        curr_row += 1
        min_dist = float("inf")
        index = 0
        curr_centroid = -1
        for centroid in centroids:
            curr_centroid += 1
            dist = np.linalg.norm(row - centroid)
            if (dist < min_dist):
                min_dist = dist
                index = curr_centroid
        index_list[index].append(curr_row)

    return index_list

"""
Outputs the observations used in this program current run aswell as the labels that were given by the KMeans
and Spectral Clustering Algorithms
"""
def output_data(observations, kmeans_labels, spectral_labels, clusters_num):
    with open("data.txt", "w") as f:
        for observation in observations:
            for cord in observation:
                f.write(str(cord))
            f.write("\n")
    f.close()

    with open("clusters.txt", "w") as f:
        f.write(str(clusters_num) + "\n")
        for labels in spectral_labels:
            f.write(str(labels)[1: -1] + "\n")
        for labels in kmeans_labels:
            f.write(str(labels)[1: -1] + "\n")
    f.close()

k_for_blobs = 5
#example
observations, labels = make_blobs(n_samples=50, n_features=3, centers=k_for_blobs)

# Main
laplace_matrix = normalized_laplacian(observations)

q = qr_decomposition(laplace_matrix)

t = tmatrix(q)
k = t.shape[1]

#g = kmeanspp.calc(observations.shape[0], num_of_clusters, t[0:].tolist(), t[0:num_of_clusters].tolist(), num_of_clusters, 300)
spectral_result = k_means_pp(k, observations.shape[0], k, 300, t[0:])
kmeans_result = k_means_pp(k, observations.shape[0], observations.shape[1], 300, observations)

spectral_clusters_array = cluster(t, spectral_result)

kmeans_clusters_array = cluster(observations, kmeans_result)

spectral_labels = visual.build_labels(len(observations), spectral_clusters_array)
kmeans_labels = visual.build_labels(len(observations), kmeans_clusters_array)

spectral_measure, kmeans_measure = visual.jaccard_measure(labels, spectral_labels, kmeans_labels)

visual.visual(observations, spectral_labels, kmeans_labels, spectral_measure, kmeans_measure, k, k_for_blobs)

output_data(observations, spectral_clusters_array, kmeans_clusters_array, k)


