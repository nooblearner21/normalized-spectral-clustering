//
// Created by Dan on 22/12/2020.
//
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "kmeans.h"


void printCentroids(int d, int k, const double* centroids)
{
    int i;
    int j;
    for(i=0; i < k; i++)
    {

        for(j=0; j < d; j++)
        {
            if(j < d - 1)
            {
                printf("%f,", *((centroids + i*d) + j));
            }
            else
            {
                printf("%f", *((centroids + i*d) + j));
            }

        }

        printf("\n");

    }
}


int getClosestClusterIndex(int k, int dimensions, const double *centroids, const double *vector)
{
    int index = 0;
    double shortestDistance;

    /*iterate through all observations(vectors)
    //for each vector iterate through all centroids
    */
    int c;
    int d;
    for(c=0; c < k; c++)
    {

        double distance = 0;
        /*for each centroid calculate the distance and save shortest distance*/
        for(d=0; d < dimensions; d++)
        {
            distance += (*((vector + d)) - (*((centroids + c*dimensions) + d))) * (*((vector + d))- *((centroids + c*dimensions) + d));
        }

        if(c == 0 || distance < shortestDistance)
        {
            index = c;
            shortestDistance = distance;
        }


    }
    /*assign the vector j the index of the closest centroid*/




    return index;
}


static PyObject* calc(PyObject *self, PyObject *args)
{

    //declare variables to parse to from API request
    PyObject *vectorList, *singleVector, *singleCord, *centroidsList;
    int vectorNum;
    int clusterNum;
    int dimensions;
    int iterations;

    Py_ssize_t vectorSize;










    int i;
    int j;
    int k;
    int d ;
    int p ;

    int amountOfVectors;

    int *vectorClusterIndices;
    double *sumOfVectors;

    int centroidChanged = 0;

    //parse arguments

    if(!PyArg_ParseTuple(args, "iiOOii:calc", &vectorNum, &dimensions, &vectorList, &centroidsList, &clusterNum, &iterations))
    {
        return NULL;
    }

    if(!PyList_Check(vectorList))
    {
        return NULL;
    }

    //declare variables for python list of lists
    PyObject *centroid_list = PyList_New(0);
    assert(centroid_list != NULL);

    PyObject *centroid_item;

    //create vectors array
    double **vectors = calloc(vectorNum, sizeof(double *));
    double *vectorPointers = calloc(vectorNum*dimensions, sizeof(double));

    //assert vector space was created
    assert(vectors != NULL);
    assert(vectorPointers != NULL);

    for(i=0; i < vectorNum; i++)
    {
        vectors[i] = vectorPointers + i*dimensions;
    }

    for(k=0; k < vectorNum; k++)
    {
        singleVector = PyList_GetItem(vectorList, k);

        if (!PyList_Check(singleVector))
        {
            continue;
        }
        vectorSize = PyList_Size(singleVector);

        for(j=0; j < vectorSize; j++)
        {
            singleCord = PyList_GetItem(singleVector, j);
            vectors[k][j] = PyFloat_AsDouble(singleCord);

        }

    }

    //create centroids array
    double **centroids = calloc(clusterNum, sizeof(double *));
    double *centroidPointer = calloc(clusterNum*dimensions, sizeof(double));

    //assert space we created properly
    assert(centroids != NULL);
    assert(centroidPointer != NULL);



    for(i=0; i < clusterNum; i++)
    {
        centroids[i] = centroidPointer + i*dimensions;
    }

    for(k=0; k < clusterNum; k++)
    {
        singleVector = PyList_GetItem(centroidsList, k);

        if (!PyList_Check(singleVector))
        {
            continue;
        }
        vectorSize = PyList_Size(singleVector);

        for(j=0; j < vectorSize; j++)
        {
            singleCord = PyList_GetItem(singleVector, j);
            centroids[k][j] = PyFloat_AsDouble(singleCord);

        }
    }




    /*clustering*/
    for(i=0; i < iterations; i++)
    {

        vectorClusterIndices = (int *)calloc(vectorNum, sizeof(int));
        assert(vectorClusterIndices != NULL);


        /*iterate through each observation for each centroid*/
        for(j=0; j < vectorNum; j++)
        {
            vectorClusterIndices[j] = getClosestClusterIndex(clusterNum, dimensions, *centroids, vectors[j]);
        }


        /*update all centroids one-by-one*/
        for(k=0; k < clusterNum; k++)
        {

            sumOfVectors = calloc(dimensions, sizeof(double));
            assert(sumOfVectors != NULL);

            /*initialize all vectors to 0*/
            for(d=0; d < dimensions; d++)
            {
                sumOfVectors[d] = 0;

            }

            amountOfVectors = 0;
            for(p=0; p < vectorNum; p++)
            {
                /*if the cluster index associated with current vector is the index of the current cluster we proceed*/
                if(vectorClusterIndices[p] == k)
                {
                    amountOfVectors += 1;
                    for(d=0; d<dimensions; d++)
                    {


                        sumOfVectors[d] += vectors[p][d];


                    }
                }
            }
            for(d=0; d<dimensions; d++)
            {
                /*check if centroid value changed to know if we can continue*/
                if(centroids[k][d] != sumOfVectors[d] / amountOfVectors)
                {
                    centroidChanged = 1;
                }

                if(amountOfVectors == 0)
                {

                    centroids[k][d] = centroids[k][d];
                }
                else
                {
                    centroids[k][d] = sumOfVectors[d] / amountOfVectors;
                }

            }
            free(sumOfVectors);
        }
        free(vectorClusterIndices);

        /*if none of the centroids changed we exit the loop*/
        if(centroidChanged)
        {
            centroidChanged = 0;
        }
        else {
            break;
        }
    }


    for(i=0; i < clusterNum; i++)
    {
        centroid_item = PyTuple_New(dimensions);
        assert(centroid_item != NULL);

        for(j=0; j < dimensions; j++) {

            PyTuple_SET_ITEM(centroid_item, j,  PyFloat_FromDouble(centroids[i][j]) );
        }

        PyList_Append(centroid_list, centroid_item);
        Py_DecRef(centroid_item);

    }


    //memory cleaning
    free(centroids);
    free(centroidPointer);
    free(vectors);
    free(vectorPointers);

    return Py_BuildValue("O", centroid_list);
}


static PyMethodDef kmeansMethods[] = {
        {"calc", (PyCFunction) calc, METH_VARARGS, PyDoc_STR("Cluster Data Using K-Means++")},
        {NULL, NULL, 0, NULL},
};

static struct PyModuleDef myModule = {
        PyModuleDef_HEAD_INIT,
        "kmeans",
        NULL,
        -1,
        kmeansMethods,
};

PyMODINIT_FUNC
PyInit_kmeanspp(void)
{
    PyObject *m;
    m = PyModule_Create(&myModule);
    if(!m)
    {
        return NULL;
    }
    return m;
}


