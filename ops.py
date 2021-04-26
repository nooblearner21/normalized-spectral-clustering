import math

import numpy as np
from numpy import linalg as LA



EPSILON = 0.0001

"""
Create the NxN weighted adjacency matrix from N vector observations
"""
def weighted_adj_matrix(observations):

    # Create a zero-value NxN matrix and initialize counters
    zeroshape = observations.shape[0]
    adj_matrix = np.zeros((zeroshape, zeroshape), dtype=np.float64)

    i = 0
    j = 0

    # Calculate euclidean distance between all nodes and fill node weights in our adjacency matrix
    for node_x in observations:
        for node_y in observations:
            dist = math.sqrt(np.sum((node_x - node_y) ** 2))
            adj_matrix[i, j] = np.exp(-dist / 2)

            j += 1
        i += 1
        j = 0

    # Change the diagonal values to zero since we do not allow edges from a node onto itself
    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix


"""
Creates the diagonal degree matrix
"""
def diagonal_degree_matrix(adj_matrix):
    diag_values = np.sum(adj_matrix, axis=1)
    diag_values_normalized = np.power(diag_values, -0.5)

    matrix_d = np.diag(diag_values_normalized)

    return matrix_d


"""
Calculates the  NxN normalized laplacian matrix given N observations
"""
def normalized_laplacian(observations):
    # Creates the weighted adjacency matrix
    adj_matrix = weighted_adj_matrix(observations)
    # Create NxN Identity Matrix I
    id_matrix = np.identity(adj_matrix.shape[1], dtype=np.float32)
    # Retrieve Diagonal degree matrix
    diagonal_matrix = diagonal_degree_matrix(adj_matrix)

    # Calculate and return laplacian
    laplacian = id_matrix - (diagonal_matrix @ adj_matrix @ diagonal_matrix)

    return laplacian


"""
Uses the QR Decomposition algortihim to calculate the eigenvalues and the corresponding
eigenvectors of a given matrix
"""

def qr_decomposition(matrix):
    n = matrix.shape[0]
    aroof = np.copy(matrix)
    qroof = np.identity(n)
    
    for i in range(n):
        q, r = mgs_algorithm(aroof)
        aroof = r @ q
        matrix_distance = np.abs(np.abs(qroof) - (np.abs(qroof @ q)))

        if (matrix_distance < EPSILON).all():
            return (aroof, qroof)

        qroof = qroof @ q

    return (aroof, qroof)


"""
Implementation of the The Modified Gram-Schmidt Algorithm used to decompose a matrix to it's eigenvalues and eigenvectors
"""
def mgs_algorithm(aroof):
    n = aroof.shape[0]
    q_matrix = np.zeros(shape=(n, n), dtype=np.float32)
    r_matrix = np.zeros(shape=(n, n), dtype=np.float32)

    for i in range(n):
        col_norm = LA.norm(aroof, axis=0)[i]
        r_matrix[i, i] = col_norm

        

        if col_norm > 0:
            q_col = np.divide(aroof[:, i], col_norm)
            q_matrix[:, i] = q_col
        else:
            raise Exception("norm is 0, so we quit the program")

        r_matrix[i, i + 1:n] = q_col.T @ aroof[:, i + 1:n]  
        aroof[:, i + 1:n] -= r_matrix[None, i, i + 1:n] * q_col.T[:, None]

    return q_matrix, r_matrix


"""
The eigengap heruistic which helps us in determining an optimal number of clusters
"""
def eigengap_heuristic(arr):
    sorted_arr = np.sort(arr)
    half_of_n = int(np.ceil(len(arr) / 2))

    k = np.argmax(np.diff(sorted_arr[:half_of_n])) + 1

    return k


"""
Creates The U matrix which is an NxK matrix that its columns represent the K eigenvectors of the smallest K eigenvalues of a matrix 
"""
def umatrix(matrix_tuple, K ,random):
    eigenvalues = matrix_tuple[0].diagonal().copy()
    eigenvectors = matrix_tuple[1].copy()

    if random:
        k = eigengap_heuristic(eigenvalues)
    else:
        k = K

    eig_index = np.argsort(eigenvalues)[:k]
    u_matrix = eigenvectors[:, eig_index]

    return u_matrix


"""
Creates the T Matrix which normalizes the rows of a given matrix
"""
def tmatrix(matrix_tuple, K, random):
    u_matrix = umatrix(matrix_tuple, K, random)
    u_matrix_rows_norms = LA.norm(u_matrix, axis=1)[:, None]

    # Normalizing each row in t_matrix
    t_matrix = np.divide(u_matrix, u_matrix_rows_norms, out=np.zeros_like(u_matrix), where=u_matrix_rows_norms != 0)

    return t_matrix


