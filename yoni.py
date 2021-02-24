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
            aroof[:, j] = aroof[:, j] - (r_matrix[i, j] @ q_matrix[:, i])

    return tuple(q_matrix, r_matrix)




#testing
test = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
print(test)

test_d = build_d_matrix(test)
print(test_d)

laplace = build_laplace_matrix(test, test_d)
print(laplace)

mgs_algorithm(laplace)
