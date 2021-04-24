import numpy as np

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

