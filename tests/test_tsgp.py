# working directory
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from optimizer.thompson_sampling import ThompsonSampling, MultiTaskThompsonSampling

# importing necessary modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm
from time import time

import argparse

# turning off automatic plot showing, and setting style
plt.ioff()
plt.style.use('fivethirtyeight')


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

# function to create an animation with the visualization
def ts_gp_animation_ackley(ts_gp, max_rounds, X, Y, output_dir=None):
    
    # dict for accumulating runs
    round_dict = {}
    
    # loop for each round
    start_time = time()
    for round_id in range(max_rounds):
        
        # recording all the info
        X_observed, y_observed = ts_gp.choose_next_sample()

        # adding to dict
        round_dict[round_id] = {'X_observed': X_observed.numpy()[:, 0], 
                                'y_observed': y_observed.numpy()
                                }
    print(f"Time Elapsed: {time()-start_time:.4f} s")
    xval, fval = ts_gp.get_result()
    print(f"Found minimum objective {fval:.4f} at {xval}")
    # plotting!
    fig, ax = plt.subplots(figsize=[10,4], dpi=150)

    # function for updating
    def animate(i):
        ax.clear()
        ax.plot(X, Y, 'k--', linewidth=2, label='Actual function')
        ax.plot(round_dict[i]['X_observed'], round_dict[i]['y_observed'], 'bo', label="""GP-Chosen Samples of Ackley's function""", alpha=0.7)
        plt.title("""Ackley's function at $y=0$, TS-GP optimization""", fontsize=14) 
        plt.xlabel('$x$') 
        plt.ylabel('$f(x)$')
        ax.set_ylim(-0.5, 10)
        return ()

    # function for creating animation
    anim = FuncAnimation(fig, animate, frames=max_rounds, interval=500, blit=True)
    if output_dir is not None:
        anim.save(f"{output_dir}/test_tsgp.mp4")
    else:
        # showing
        return HTML(anim.to_html5_video())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1, required=False, choices=[1, 2])
    parser.add_argument("--step", type=int, default=50, required=False)
    args = parser.parse_args()

    # ground truth data for 1d ackley
    X = np.linspace(-4, 4, 500)
    Y = ackley_1d(X)

    print(f"Testing bayesian optimization on {args.dim}-D Ackley function...")
    if not args.dim==2:
        bounds = np.array([-4, 4])
        objective = ackley_1d
    else:
        bounds = np.array([[-4, 4], [-4, 4]])
        objective = ackley_2d

    # instance of multi-task TS-GP
    multi_ts_gp = MultiTaskThompsonSampling(n_random_draws=5, objective=objective, bounds=bounds)
    
    # showing animnation
    ts_gp_animation_ackley(multi_ts_gp, args.step, X, Y, "figures/")
    print("The result video has been saved under figures folder")