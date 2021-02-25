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




def cluster(matrix_t, centroids):

    #create list of K lists
    index_list = [[] for i in range(len(centroids))]

    curr_row = -1
    for row in matrix_t:
        curr_row += 1
        min_dist = float("inf")
        index = 0
        curr_centroid = -1
        for centroid in centroids:
            curr_centroid += 1
            dist = numpy.linalg.norm(row-centroid)
            if(dist < min_dist):
                min_dist = dist
                index = curr_centroid
        index_list[index].append(curr_row)
    
    return index_list

def output_data(index_list, cluster_num):
    with open file("clusters.txt", "w") as f:
        f.write(cluster_num)
        for indices in index_list:
            if len(indices) == 0:
                break
            else:
                f.write(indices)
    f.close()
    

