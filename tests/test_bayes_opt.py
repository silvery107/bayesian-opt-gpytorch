# working directory
import argparse
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from models.bayesian_optimization import BayesianOptimization
import numpy as np
import torch
from time import time

# defining the function, y=0 to get a 1D cut at the origin
def ackley_1d(x, y=0):
    
    # the formula is rather large
    out = (-20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) 
           - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
           + np.e + 20)
    
    # returning
    return out

def ackley_2d(X):
    out = ackley_1d(X[0], X[1])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1, required=False, choices=[1, 2])
    parser.add_argument("--warmup", type=int, default=5, required=False)
    parser.add_argument("--step", type=int, default=50, required=False)
    args = parser.parse_args()

    assert args.warmup < args.step
    if args.dim == 1:
        lower = torch.tensor(-4)
        upper = torch.tensor(4)
        objective = ackley_1d
    else:
        lower = torch.tensor([-4, -4])
        upper = torch.tensor([4, 4])
        objective = ackley_2d
        
    optimizer = BayesianOptimization(lower, upper, n_warmup=args.warmup)

    start_time = time()
    xval, fval = optimizer.minimize(objective, args.step)
    print(f"Time Elapsed: {time()-start_time:.4f} s")
    print(f"Found minimum objective {fval:.4f} at {xval}")

    start_time = time()
    xval, fval = optimizer.minimize_tsgp(objective, args.step-args.warmup+1)
    print(f"Time Elapsed: {time()-start_time:.4f} s")
    print(f"Found minimum objective {fval:.4f} at {xval}")