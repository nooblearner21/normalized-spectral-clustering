import math

import numpy as np
from numpy import linalg as LA


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
            adj_matrix[i, j] = np.exp(-dist) / 2

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

    #Creates the weighted adjacency matrix
    adj_matrix = weighted_adj_matrix(observations)
    # create NxN Identity Matrix I
    id_matrix = np.identity(adj_matrix.shape[1])
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

    aroof = np.copy(matrix)
    qroof = np.identity(len(matrix))

    for i in range(len(matrix)):
        q, r = mgs_algorithm(aroof)
        aroof = r @ q
        matrix_distance = np.abs(np.abs(qroof) - (np.abs(qroof @ q)))

        if((matrix_distance < 0.0001).all()):
            return (aroof, qroof)

        qroof = qroof @ q

    return (aroof, qroof)


"""
Implementation of the The Modified Gram-Schmidt Algorithm used to decompose a matrix to it's eigenvalues and eigenvectors
"""
def mgs_algorithm(aroof):
    # avoiding rounded values because of int ndarray
    aroof = aroof.astype(float)

    # adding identity matrix to aroof to avoid columns with norm 0
    # aroof = aroof + np.identity(aroof.shape[0])

    n = len(aroof)
    q_matrix = np.zeros(shape=(n, n))
    r_matrix = np.zeros(shape=(n, n))

    for i in range(n):

        column_i_norm = LA.norm(aroof, axis=0)[i]

        r_matrix[i, i] = column_i_norm

        if column_i_norm > 0:
            q_matrix[:, i] = np.divide(aroof[:, i], column_i_norm)
        else:
            raise Exception("norm is 0, so we quit the program")

        q_i_column_transpose = q_matrix[:, i].T
        aroof_j_columns = np.copy(aroof[:, i + 1:n])
        r_row_i_j_values = q_i_column_transpose @ aroof_j_columns
        r_matrix[i, i + 1:n] = r_row_i_j_values

        r_ij_q_i = q_i_column_transpose[:, None] * r_row_i_j_values.reshape(1, r_row_i_j_values.shape[0])
        aroof_j_new_columns = aroof_j_columns - r_ij_q_i
        aroof[:, i + 1:n] = aroof_j_new_columns

    return (q_matrix, r_matrix)


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
def umatrix(matrix_tuple):
    n = len(matrix_tuple[0])

    eigenvalues = matrix_tuple[0].diagonal().copy()
    eigenvectors = matrix_tuple[1].copy()

    k = eigengap_heuristic(eigenvalues)

    eig_index = np.argsort(eigenvalues)[:k]
    u_matrix = eigenvectors[:, eig_index]

    return u_matrix

"""
Creates the T Matrix which normalizes the rows of a given matrix
"""
def tmatrix(matrix_tuple):
    u_matrix = umatrix(matrix_tuple)

    n = len(u_matrix)
    k = len(u_matrix[0])

    u_matrix_rows_norms = LA.norm(u_matrix, axis=1)[:, None]

    t_matrix = np.divide(u_matrix, u_matrix_rows_norms, out=np.zeros_like(u_matrix), where=u_matrix_rows_norms != 0)

    return t_matrix


