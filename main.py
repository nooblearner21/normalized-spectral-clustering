import argparse
from sklearn.datasets import make_blobs

parser = argparse.ArgumentParser()
parser.add_argument("k", type=int)
parser.add_argument("n", type=int)
parser.add_argument('--random', dest='random', action='store_true')
parser.add_argument('--no-random', dest='random', action='store_false')

args = parser.parse_args()

k = args.k
n = args.n
random = args.random

print("\n\n\n\n")
if(random):
    x = make_blobs(n_samples=4)
    print(x[0])

print("\n\n\n\n")