import numpy as np
from numpy import linalg as LA


def build_d_matrix(adj_matrix):
    diag_values = np.sum(adj_matrix, axis=1)
    diag_values_normalized = np.power(diag_values, -0.5)

    matrix_d = np.diag(diag_values_normalized)

    return matrix_d


def build_laplace_matrix(adj_matrix, d_matrix):
    n = len(d_matrix)
    return np.identity(n) - d_matrix @ adj_matrix @ d_matrix


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

    for i in range(0, int(np.floor(n / 2))):
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
        t_matrix[i] = np.divide(u_matrix[i], u_matrix_rows_norms[i])

    return t_matrix



#testing
test = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
test_2 = np.array([[5, 5, 5, 5], [2, 2, 2, 2], [1, 1, 1, 1], [4, 4, 4, 4]])

test_d = build_d_matrix(test)

laplace = build_laplace_matrix(test, test_d)
