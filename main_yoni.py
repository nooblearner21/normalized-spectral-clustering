import numpy as np

#We want to build D by taking W and calculating the degree of each point


def build_diagonal_matrix(adj_matrix):
    diag_values = np.sum(adj_matrix, axis=1)
    diag_values_normalized = np.power(diag_values, -0.5)

    matrix_d = np.diag(diag_values_normalized)

    return matrix_d

def build_laplace_matrix(adj_matrix, d_matrix):
    return np.subtract(np.identity(len(d_matrix)), np.matmul(np.matmul(d_matrix, adj_matrix), d_matrix))
