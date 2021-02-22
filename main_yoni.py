import numpy as np

#We want to build D by taking W and calculating the degree of each point


def build_d_matrix(adj_matrix):
    diag_values = np.sum(adj_matrix, axis=1)
    diag_values_normalized = np.power(diag_values, -0.5)

    matrix_d = np.diag(diag_values_normalized)

    return matrix_d

def build_laplace_matrix(adj_matrix, d_matrix):
    return np.subtract(np.identity(len(d_matrix)), np.matmul(np.matmul(d_matrix, adj_matrix), d_matrix))


#testing
test = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
print(test)

test_d = build_d_matrix(test)
print(test_d)

print(build_laplace_matrix(test, test_d))
