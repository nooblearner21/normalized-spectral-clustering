import argparse
import numpy as np
from sklearn.datasets import make_blobs

import visual
from ops import qr_decomposition, tmatrix, normalized_laplacian
from utils import cluster, output_data

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
d = np.random.randint(2, 4)


if random:
    n = np.random.randint(50, 350)
    k = np.random.randint(5, 20)


observations, labels = make_blobs(n_samples=n, n_features=d, centers=k)


print(random)

"""
Returns the corresponding cluster index ~ observation relationship
"""
def cluster(tmatrix, centroids):

    # create an empty list of K centroids
    index_list = [[] for i in range(len(centroids))]


    #Loop through each row of tmatrix and find the closest centroid
    #to the coordinates of that row
    curr_row = -1
    for row in tmatrix:
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
def output_data(observations, blob_labels, kmeans_labels, spectral_labels, clusters_num):
    with open("data.txt", "w") as f:
        label = 0
        for observation in observations:
            for cord in observation:
                f.write(str(cord) + ",")
            f.write(str(blob_labels[label]))
            label += 1
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

t = tmatrix(q, k, random)

# Saving the original k in case random == true
k_for_blobs = k

k = t.shape[1]

# Calculating the centroids from both algorithms
spectral_result = k_means_pp(k, observations.shape[0], k, 300, t[0:])
kmeans_result = k_means_pp(k, observations.shape[0], observations.shape[1], 300, observations)


# Clustering the observations according to the results
spectral_clusters_array = cluster(t, spectral_result)
kmeans_clusters_array = cluster(observations, kmeans_result)

# Building the labels for matplotlib
spectral_labels = visual.build_labels(len(observations), spectral_clusters_array)
kmeans_labels = visual.build_labels(len(observations), kmeans_clusters_array)


# Calculating the Jaccard measures for both algorithms
spectral_measure, kmeans_measure = visual.jaccard_measure(labels, spectral_labels, kmeans_labels)


visual.visual(observations, spectral_labels, kmeans_labels, spectral_measure, kmeans_measure, k, k_for_blobs)

output_data(observations, labels, spectral_clusters_array, kmeans_clusters_array, k)


