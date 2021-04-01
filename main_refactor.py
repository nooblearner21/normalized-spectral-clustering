import argparse
import numpy as np

import laplace
import build_t_matrix

import kmeanspp

parser = argparse.ArgumentParser()
parser.add_argument("k", type=int)
parser.add_argument("n", type=int)
parser.add_argument('--random', dest='random', action='store_true')
parser.add_argument('--no-random', dest='random', action='store_false')

args = parser.parse_args()

k = args.k
n = args.n
random = args.random

#example
observations = np.zeros(shape=10)

# Main
laplace_matrix = laplace.get_normalized_laplacian(observations)

q = qr_iterations(z)

t = build_t_matrix(q)

g = kmeanspp.calc(x.shape[0], t.shape[1], list(x), list(x), t.shape[1], 300)