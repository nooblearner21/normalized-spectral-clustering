import numpy as np


#Receive N observations ndarray as argument and returns the
#weighted adjacency matrix
def get_weighted_adj_matrix(observations):
    
    #Create a zero-value NxN matrix
    weightedMatrix = np.zeros_like(observations, dtype=np.float64)

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


x = np.arange(1, 17).reshape(4,4)




def getNormalizedLaplacian(weightedAdjMatrix):
    
    #create NxN Identity Matrix I
    idMatrix = np.identity(weightedAdjMatrix.shape[1])
    #Retrieve Diagonal degree matrix
    diagonalMatrix = getDiagonalDegreeMatrix(weightedAdjMatrix)

    #Calculate and return laplacian
    laplacian = idMatrix - (diagonalMatrix @ weightedAdjMatrix @ diagonalMatrix)

    return laplacian
    
getNormalizedLaplacian(getWeightedMatrix(x))



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
