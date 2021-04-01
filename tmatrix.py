import numpy as np
from numpy import linalg as LA


def qr_iterations(matrix):
    aroof = np.copy(matrix)
    qroof = np.identity(matrix.shape[1])

    # change to N
    for i in range(1000):
        qr = mgs_algorithm(aroof)

        aroof = qr[1] @ qr[0]

        matrix_distance = np.abs(np.abs(qroof) - (np.abs(qroof @ qr[0])))

        if ((matrix_distance < 0.0001).all()):
            return (aroof, qroof)
        qroof = qroof @ qr[0]

    return (aroof, qroof)


def mgs_algorithm(aroof):
    n = len(aroof)
    q_matrix = np.zeros(shape=(n, n))
    r_matrix = np.zeros(shape=(n, n))

    for i in range(n):

        column_i_norm = LA.norm(aroof, axis=0)[i]

        r_matrix[i, i] = column_i_norm

        if column_i_norm > 0.0000001:

            q_matrix[:, i] = np.divide(aroof[:, i], column_i_norm)
        else:
            raise Exception("norm is 0, so we quit the program")

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

    # k + 1 because arrays starts from 0...
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

    # change to numpy function instead of for loop to improve performance
    for i in range(n):
        row_i_norm = u_matrix_rows_norms[i]
        if row_i_norm != 0:
            t_matrix[i] = np.divide(u_matrix[i], u_matrix_rows_norms[i])

    return t_matrix


def t_matrix(laplace):
    return build_t_matrix(qr_iterations(laplace))


