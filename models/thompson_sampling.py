import torch
import gpytorch
import numpy as np
from models.gp_models import Matern_GP, train_gp_hyperparams


# TS-GP optimizer
class ThompsonSamplingGP:
    
    # initialization
    def __init__(self, n_random_draws, objective, x_bounds, interval_resolution=1000, noise_level=1e-3):

        # number of random samples before starting the optimization
        self.n_random_draws = n_random_draws
        
        # the objective is the function we're trying to optimize
        self.objective = objective
        
        # the bounds tell us the interval of x we can work
        self.bounds = x_bounds
        
        # interval resolution is defined as how many points we will use to 
        # represent the posterior sample
        # we also define the x grid
        self.interval_resolution = interval_resolution
        self.X_grid = np.linspace(self.bounds[0], self.bounds[1], self.interval_resolution)
        
        # also initializing our design matrix and target variable
        self.X = np.array([])
        self.y = np.array([])

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.initialize(noise=noise_level)
        self.gp = None
    
    # process of choosing next point
    def choose_next_sample(self):
        
        # if we do not have enough samples, sample randomly from bounds
        if self.X.shape[0] < self.n_random_draws:
            next_sample = np.random.uniform(self.bounds[0], self.bounds[1],1)[0]
        
        # if we do, we fit the GP and choose the next point based on the posterior draw minimum
        else:
            # 1. Fit the GP to the observations we have
            train_x = torch.from_numpy(self.X)
            train_y = torch.from_numpy(self.y)
            if self.gp is None:
                self.gp = Matern_GP(train_x, train_y)
            else:
                self.gp.set_train_data(train_x, train_y, strict=False)
            
            # train the GP
            train_gp_hyperparams(self.gp, self.likelihood, train_x, train_y, lr=0.1, iters=50)
            
            self.gp.eval().double()
            self.likelihood.eval().double()
            
            # 2. Draw one sample (a function) from the posterior
            input_x = torch.from_numpy(self.X_grid)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_obs = self.likelihood(self.gp(input_x))
                posterior_sample = pred_obs.sample().numpy()

            # 3. Choose next point as the optimum of the sample
            which_min = np.argmin(posterior_sample)
            next_sample = self.X_grid[which_min]
        
            # let us also get the std from the posterior, for visualization purposes
            posterior_mean = pred_obs.mean.numpy()
            posterior_std = pred_obs.stddev.numpy()
        
        # let us observe the objective and append this new data to our X and y
        next_observation = self.objective(next_sample)
        self.X = np.append(self.X, next_sample)
        self.y = np.append(self.y, next_observation)
        
        # return everything if possible
        try:
            # returning values of interest
            return self.X, self.y, self.X_grid, posterior_sample, posterior_mean, posterior_std
        
        # if not, return whats possible to return
        except:
            return (self.X, self.y, self.X_grid, np.array([np.mean(self.y)]*self.interval_resolution), 
                    np.array([np.mean(self.y)]*self.interval_resolution), np.array([0]*self.interval_resolution))