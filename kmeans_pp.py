import argparse
import numpy as np
import kmeanspp


def k_means_pp(K, N, d, MAX_ITER, observations):
    np.random.seed(0)

    centroids = np.zeros(shape=(K, d), dtype=float)
    centroids_indexes = np.zeros(shape=K, dtype=int)

    first_chosen_centroid_index = np.random.choice(N)
    centroids[0] = observations[first_chosen_centroid_index]
    centroids_indexes[0] = first_chosen_centroid_index

    min_distances = np.power(observations - centroids[0], 2).sum(axis=1)

    for j in range(1, K):
        curr_min_distances = np.power(observations - centroids[j - 1], 2).sum(axis=1)
        min_distances = np.minimum(min_distances, curr_min_distances)

        omega = np.sum(min_distances)
        prob_array = np.divide(min_distances, omega)

        chosen_centroid_index = np.random.choice(N, p=prob_array)
        centroids[j] = observations[chosen_centroid_index]
        centroids_indexes[j] = chosen_centroid_index

    for i in range(0, K-1):
        print(centroids_indexes[i], end=',')
    print(centroids_indexes[K-1])
    return kmeanspp.calc(N, d, observations.tolist(), centroids.tolist(), K, MAX_ITER)
