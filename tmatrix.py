import numpy as np
from numpy import linalg as LA


def qr_iterations(matrix):
    aroof = np.copy(matrix)
    qroof = np.identity(matrix.shape[1])

    N = len(matrix)
    # change to N
    for i in range(N):
        qr = mgs_algorithm(aroof)

        aroof = qr[1] @ qr[0]
        matrix_distance = np.abs(np.abs(qroof) - (np.abs(qroof @ qr[0])))

        if ((matrix_distance < 0.0001).all()):
            return (aroof, qroof)
        qroof = qroof @ qr[0]

    return (aroof, qroof)


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

    u_matrix_rows_norms = LA.norm(u_matrix, axis=1)[:, None]

    t_matrix = np.divide(u_matrix, u_matrix_rows_norms, out=np.zeros_like(u_matrix), where=u_matrix_rows_norms != 0)

    return t_matrix


def t_matrix(laplace):
    return build_t_matrix(qr_iterations(laplace))

