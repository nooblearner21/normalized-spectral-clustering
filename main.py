import argparse
import numpy as np
from sklearn.datasets import make_blobs

from visual import visual, build_labels, jaccard_measure
from ops import qr_decomposition, tmatrix, normalized_laplacian
from utils import cluster, output_data

from kmeans_pp import k_means_pp

parser = argparse.ArgumentParser()
parser.add_argument("k", type=int)
parser.add_argument("n", type=int)
parser.add_argument('--Random', dest='random', action='store_true')
parser.add_argument('--no-Random', dest='random', action='store_false')

args = parser.parse_args()

K = args.k
n = args.n
random = args.random
d = np.random.randint(2, 4)
MAX_ITER = 300

if random:
    n = np.random.randint(50, 350)
    K = np.random.randint(5, 20)

# Validation of input
if n < K:
    raise Exception("n can't be smaller than k")
if n <= 0 or K <= 0:
    raise Exception("n and k can't be non-positive")

# Generating the observations
observations, labels = make_blobs(n_samples=n, n_features=d, centers=K, random_state=0)



# The Normalized Spectral Clustering Algorithm
laplace_matrix = normalized_laplacian(observations)
q = qr_decomposition(laplace_matrix)
t = tmatrix(q, K, random)
k = t.shape[1]

# Calculating the centroids from both algorithms
spectral_result = k_means_pp(k, n, k, MAX_ITER, t[0:])
kmeans_result = k_means_pp(k, n, d, MAX_ITER, observations)

# Clustering the observations according to the results
spectral_clusters_array = cluster(t, spectral_result)
kmeans_clusters_array = cluster(observations, kmeans_result)

# Building the labels for matplotlib
spectral_labels = build_labels(n, spectral_clusters_array)
kmeans_labels = build_labels(n, kmeans_clusters_array)

# Calculating the Jaccard measures for both algorithms
spectral_measure, kmeans_measure = jaccard_measure(labels, spectral_labels, kmeans_labels)

# Output
visual(observations, spectral_labels, kmeans_labels, spectral_measure, kmeans_measure, k, K)
output_data(observations, labels, spectral_clusters_array, kmeans_clusters_array, k)


