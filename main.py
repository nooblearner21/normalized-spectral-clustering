import argparse
from sklearn.datasets import make_blobs
from scipy.sparse.csgraph import laplacian
import numpy as np
from numpy import linalg as LA
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


# x = make_blobs(n_samples=4)
x = [[1,1], [2,2], [3,3], [4,4]]
x = np.array(x)




#Receive N observations ndarray as argument and returns the
#weighted adjacency matrix
def get_weighted_adj_matrix(observations):
    
    #Create a zero-value NxN matrix

    zeroshape = observations.shape[0]
    weightedMatrix = np.zeros((zeroshape,zeroshape), dtype=np.float64)
    print(weightedMatrix)

    #counters
    i = 0
    j = 0

    #Calculate euclidean distance between all nodes and fill node weights in our adjacency matrix
    for vectorx in observations:
        for vectory in observations:
            dist = np.sum((vectorx - vectory) ** 2) / 2
            weightedMatrix[i,j] = np.exp(-dist)

            j += 1
        i += 1
        j = 0
    
    #Change the diagonal values to zero since we do not allow edges from a node to itself
    np.fill_diagonal(weightedMatrix, 0)
    
    return weightedMatrix



def get_diagonal_degree_matrix(adj_matrix):
    diag_values = np.sum(adj_matrix, axis=1)
    diag_values_normalized = np.power(diag_values, -0.5)

    matrix_d = np.diag(diag_values_normalized)

    return matrix_d


def get_normalized_laplacian(weighted_adj_matrix):
    
    #create NxN Identity Matrix I
    id_matrix = np.identity(weighted_adj_matrix.shape[1])
    #Retrieve Diagonal degree matrix
    diagonal_matrix = get_diagonal_degree_matrix(weighted_adj_matrix)

    #Calculate and return laplacian
    laplacian = id_matrix - (diagonal_matrix @ weighted_adj_matrix @ diagonal_matrix)

    return laplacian


def qr_iterations(matrix):
    aroof = np.copy(matrix)
    qroof = np.identity(matrix.shape[1])

    #change to N
    for i in range(10000):
        qr = mgs_algorithm(aroof)
        
        aroof = qr[1] @ qr[0]


        matrix_distance = np.abs(np.abs(qroof) - (np.abs(qroof @ qr[0])))

        if((matrix_distance > 0.0001).all()):
            return (aroof, qroof)
        qroof = qroof @ qr[0]
    
    return (aroof, qroof)

def mgs_algorithm(aroof):
    n = len(aroof)
    q_matrix = np.zeros(shape=(n, n))
    r_matrix = np.zeros(shape=(n, n))

    for i in range(n):
        column_i_norm = LA.norm(aroof, axis=0)[i]

        r_matrix[i, i] = np.power(column_i_norm, 2)
        q_matrix[:, i] = np.divide(aroof[:, i], column_i_norm)

        for j in range(i + 1, n):
            r_matrix[i, j] = q_matrix.transpose()[:, i] @ aroof[:, j]
            aroof[:, j] = aroof[:, j] - (r_matrix[i, j] * q_matrix[:, i])

    return (q_matrix, r_matrix)


def eigengap_heuristic(arr):
    n = len(arr)
    sorted_arr = sorted(arr)
    k = 0
    max_gap = 0

    for i in range(0, int(np.ceil(n / 2))):
        curr_gap = np.abs(sorted_arr[i] - sorted_arr[i + 1])

        if max_gap < curr_gap:
            max_gap = curr_gap
            k = i

    return k


def build_u_matrix(matrix_tuple):
    n = len(matrix_tuple[0])
    eigenvalues = matrix_tuple[0].diagonal().copy()
    eigenvectors = matrix_tuple[1].transpose().copy()
    k = eigengap_heuristic(eigenvalues) + 1

    eigen_map = []

    for i in range(n):
        eigen_map.append({eigenvalues[i]: eigenvectors[i]})

    sorted_map = sorted(eigen_map, key=lambda item: list(item.keys())[0])

    u_matrix = np.ndarray(shape=(n, k))

    for i in range(k):
        vector = list(sorted_map[i].values())[0]
        u_matrix[:, i] = vector

    return u_matrix


def build_t_matrix(matrix_tuple):
    u_matrix = build_u_matrix(matrix_tuple)

    n = len(u_matrix)
    k = len(u_matrix[0])
    t_matrix = np.ndarray(shape=(n, k))
    u_matrix_rows_norms = LA.norm(u_matrix, axis=1)

    for i in range(n):
        row_i_norm = u_matrix_rows_norms[i]
        if row_i_norm != 0:
            t_matrix[i] = np.divide(u_matrix[i], u_matrix_rows_norms[i])

    return t_matrix



def cluster(matrix_t, centroids):

    #create list of K lists
    index_list = [[] for i in range(len(centroids))]

    curr_row = -1
    for row in matrix_t:
        curr_row += 1
        min_dist = float("inf")
        index = 0
        curr_centroid = -1
        for centroid in centroids:
            curr_centroid += 1
            dist = numpy.linalg.norm(row-centroid)
            if(dist < min_dist):
                min_dist = dist
                index = curr_centroid
        index_list[index].append(curr_row)
    
    return index_list

def output_data(index_list, cluster_num):
    with open("clusters.txt", "w") as f:
        f.write(cluster_num)
        for indices in index_list:
            if len(indices) == 0:
                break
            else:
                f.write(indices)
    f.close()
    





if __name__ == '__main__':
    print(x)
    y = get_weighted_adj_matrix(x)
    z = get_normalized_laplacian(y)

    q = qr_iterations(z)
    t = build_t_matrix(q)


    g = kmeanspp.calc(x.shape[0], t.shape[1], list(x), list(x), t.shape[1], 300)
