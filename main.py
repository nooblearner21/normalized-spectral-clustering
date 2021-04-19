import argparse
import numpy as np
from sklearn.datasets import make_blobs

import laplace
import visual
from ops import qr_iterations, build_t_matrix

import kmeanspp

parser = argparse.ArgumentParser()
parser.add_argument("k", type=int)
parser.add_argument("n", type=int)
parser.add_argument('--random', dest='random', action='store_true')
parser.add_argument('--no-random', dest='random', action='store_false')

args = parser.parse_args()

k = args.k
n = args.n
random = args.random

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



#example
observations, labels = make_blobs(n_samples=30, n_features=3)

# Main
laplace_matrix = laplace.get_normalized_laplacian(observations)

q = qr_iterations(laplace_matrix)

t = build_t_matrix(q)
num_of_clusters = t.shape[1]

g = kmeanspp.calc(observations.shape[0], num_of_clusters, t[0:].tolist(), t[0:num_of_clusters].tolist(), num_of_clusters, 300)

clusters_array = cluster(t, g)

my_labels = visual.build_labels(len(observations), clusters_array)

visual.visual(observations, my_labels, labels)
