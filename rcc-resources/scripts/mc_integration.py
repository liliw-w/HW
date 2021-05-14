"""
compute pi using Monte Carlo integration
"""
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser(description='Compute pi using Monte Carlo Integration.')
parser.add_argument('samples', metavar='N', type=int,
                    help='number of samples')
parser.add_argument('--file', dest='fname', type=argparse.FileType('w'),
                    default="mc_pi.csv",
                    help='file to store the answer (defualt: mc_pi.csv')

args = parser.parse_args()

print("hello")
print(args.samples)


# write number of samples to file
args.fname.write("{}\n".format(args.samples))

# estimate value of pi
def compute_pi_mc(n=1000):
    x = np.random.rand(n,2)*2 - 1
    r = np.linalg.norm(x, axis=1)
    return 4 * np.sum(r < 1.) / n

t0 = time.time()
pi_est = compute_pi_mc(args.samples)
t1 = time.time()
print("{} sec. elapsed".format(t1 - t0))
print("estimated value of pi: {}".format(pi_est))

# write computed value of pi to file
args.fname.write("{}\n".format(pi_est))
