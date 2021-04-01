# Receive N observations ndarray as argument and returns the
# weighted adjacency matrix
def get_weighted_adj_matrix(observations):
    # Create a zero-value NxN matrix

    zeroshape = observations.shape[0]
    weightedMatrix = np.zeros((zeroshape, zeroshape), dtype=np.float64)

    # counters
    i = 0
    j = 0

    # Calculate euclidean distance between all nodes and fill node weights in our adjacency matrix
    for vectorx in observations:
        for vectory in observations:
            dist = math.sqrt(np.sum((vectorx - vectory) ** 2))
            weightedMatrix[i, j] = np.exp(-dist) / 2

            j += 1
        i += 1
        j = 0

    # Change the diagonal values to zero since we do not allow edges from a node to itself
    np.fill_diagonal(weightedMatrix, 0)

    return weightedMatrix


def get_diagonal_degree_matrix(adj_matrix):
    diag_values = np.sum(adj_matrix, axis=1)
    diag_values_normalized = np.power(diag_values, -0.5)

    matrix_d = np.diag(diag_values_normalized)

    return matrix_d


def get_normalized_laplacian(observations):
    weighted_adj_matrix = get_weighted_adj_matrix(observations)
    # create NxN Identity Matrix I
    id_matrix = np.identity(weighted_adj_matrix.shape[1])
    # Retrieve Diagonal degree matrix
    diagonal_matrix = get_diagonal_degree_matrix(weighted_adj_matrix)

    # Calculate and return laplacian
    laplacian = id_matrix - (diagonal_matrix @ weighted_adj_matrix @ diagonal_matrix)

    return laplacian