# working directory
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from models.thompson_sampling import ThompsonSamplingGP 

# importing necessary modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm
from time import time

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


# function to create an animation with the visualization
def ts_gp_animation_ackley(ts_gp, max_rounds, X, Y, output_dir=None):
    
    # dict for accumulating runs
    round_dict = {}
    
    # loop for each round
    start_time = time()
    for round_id in range(max_rounds):
        
        # recording all the info
        X_observed, y_observed, X_grid, posterior_sample, posterior_mean, posterior_std = ts_gp.choose_next_sample()

        
        # adding to dict
        round_dict[round_id] = {'X_observed': X_observed.numpy(), 
                                'y_observed': y_observed.numpy(), 
                                'X_grid': X_grid.numpy(),
                                'posterior_sample': posterior_sample.numpy(),
                                'posterior_mean': posterior_mean.numpy(),
                                'posterior_std': posterior_std.numpy()}
    print(f"Time Elapsed: {time()-start_time:.4f} s")
    # plotting!
    fig, ax = plt.subplots(figsize=[10,4], dpi=150)
    
    # plotting first iteration
    ax.plot(X, Y, 'k--', linewidth=2, label='Actual function')
    ax.plot(round_dict[0]['X_observed'], round_dict[0]['y_observed'], 'bo', label="""GP-Chosen Samples of Ackley's function""", alpha=0.7)
    ax.plot(round_dict[0]['X_grid'], round_dict[0]['posterior_sample'], 'r', linewidth=2, label='Sample from the posterior', alpha=0.7)
    ax.fill_between(round_dict[0]['X_grid'], round_dict[0]['posterior_mean'] - round_dict[0]['posterior_std'],
                     round_dict[0]['posterior_mean'] + round_dict[0]['posterior_std'], alpha=0.2, color='r')
    plt.title("""Ackley's function at $y=0$, TS-GP optimization""", fontsize=14) 
    plt.xlabel('$x$') 
    plt.ylabel('$f(x)$') 
    ax.set_ylim(-0.5, 10)
        
    # function for updating
    def animate(i):
        ax.clear()
        ax.plot(X, Y, 'k--', linewidth=2, label='Actual function')
        ax.plot(round_dict[i]['X_observed'], round_dict[i]['y_observed'], 'bo', label="""GP-Chosen Samples of Ackley's function""", alpha=0.7)
        ax.plot(round_dict[i]['X_grid'], round_dict[i]['posterior_sample'], 'r', linewidth=2, label='Sample from the posterior', alpha=0.7)
        ax.fill_between(round_dict[i]['X_grid'], round_dict[i]['posterior_mean'] - round_dict[i]['posterior_std'],
                         round_dict[i]['posterior_mean'] + round_dict[i]['posterior_std'], alpha=0.2, color='r')
        plt.title("""Ackley's function at $y=0$, TS-GP optimization""", fontsize=14) 
        plt.xlabel('$x$') 
        plt.ylabel('$f(x)$')
        ax.set_ylim(-0.5, 10)
        return ()

    # function for creating animation
    anim = FuncAnimation(fig, animate, frames=max_rounds, interval=2000, blit=True)
    if output_dir is not None:
        anim.save(f"{output_dir}/test_tsgp.mp4")
    else:
        # showing
        return HTML(anim.to_html5_video())

if __name__ == "__main__":
    # instance of our TS-GP
    ts_gp = ThompsonSamplingGP(n_random_draws=2, objective=ackley_1d, x_bounds=(-4,4))
    # suppressing warnings, gp can complain at times
    import warnings 
    warnings.filterwarnings("ignore")
    
    # data
    X = np.linspace(-4, 4, 500)
    Y = ackley_1d(X)
    # showing animnation
    ts_gp_animation_ackley(ts_gp, 20, X, Y, "figures/")