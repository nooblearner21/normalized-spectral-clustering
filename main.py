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

output_data(observations, spectral_clusters_array, kmeans_clusters_array, k)


