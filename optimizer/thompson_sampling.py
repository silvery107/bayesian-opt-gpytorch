import torch
import gpytorch
import numpy as np
from model.gp_models import Matern_GP, train_gp_hyperparams
from torch.distributions.uniform import Uniform

# TS-GP optimizer
class ThompsonSampling:
    
    # initialization
    def __init__(self, n_random_draws, objective, x_bounds, interval_resolution=1000, noise_level=1e-3, dtype=torch.float, device="cpu"):

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
        self.X_grid = torch.linspace(self.bounds[0], self.bounds[1], self.interval_resolution, dtype=dtype, device=device)
        
        # also initializing our design matrix and target variable
        self.X = None
        self.y = None

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.initialize(noise=noise_level).to(dtype=dtype, device=device)
        self.gp = None

        self.dtype = dtype
        self.device = device

    def warm_start(self):
        samples = []
        observations = []
        for i in range(self.n_random_draws):
            sample = (self.bounds[1] - self.bounds[0]) * torch.rand(1)[0] + self.bounds[0]
            observation = self.objective(sample)
            samples.append(sample)
            observations.append(observation)
        
        self.X = torch.tensor(samples, dtype=self.dtype, device=self.device)
        self.y = torch.tensor(observations, dtype=self.dtype, device=self.device)
    
    # process of choosing next point
    def choose_next_sample(self):
        
        # if we do not have enough samples, sample randomly from bounds
        if self.X is None:
            self.warm_start()
        
        # if we do, we fit the GP and choose the next point based on the posterior draw minimum
        else:
            # 1. Fit the GP to the observations we have
            if self.gp is None:
                self.gp = Matern_GP(self.X, self.y).to(dtype=self.dtype, device=self.device)
            else:
                self.gp.set_train_data(self.X, self.y, strict=False)
            
            # train the GP
            train_gp_hyperparams(self.gp, self.likelihood, self.X, self.y, lr=0.1, iters=50)
            
            # 2. Draw one sample (a function) from the posterior
            self.gp.eval()
            self.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_obs = self.likelihood(self.gp(self.X_grid))
                posterior_sample = pred_obs.sample()

            # 3. Choose next point as the optimum of the sample
            which_min = torch.argmin(posterior_sample)
            next_sample = self.X_grid[which_min]
        
            # get the std from the posterior, for visualization purposes
            # posterior_mean = pred_obs.mean
            # posterior_std = pred_obs.stddev
        
            # let us observe the objective and append this new data to our X and y
            next_observation = self.objective(next_sample)
            self.X = torch.cat([self.X, torch.atleast_1d(next_sample)])
            self.y = torch.cat([self.y, torch.atleast_1d(next_observation)])
        
        return self.X, self.y
        
class MultiTaskThompsonSampling:
    # initialization
    def __init__(self, n_random_draws, objective, bounds, interval_resolution=1000, noise_level=1e-3, dtype=torch.float32, device="cpu"):

        # number of random samples before starting the optimization
        self.n_random_draws = n_random_draws 
        
        # the objective is the function we're trying to optimize
        self.objective = objective
        
        # the lower and upper bounds of variables to be optimized
        if torch.is_tensor(bounds):
            self.X_bounds = bounds
        elif isinstance(bounds, np.ndarray):
            self.X_bounds = torch.from_numpy(bounds).to(dtype=dtype, device=device)
        else:
            self.X_bounds = torch.tensor(bounds, dtype=dtype, device=device) # (N, 2)
        
        if len(self.X_bounds.shape) < 2:
            self.X_bounds = self.X_bounds.unsqueeze(0)
        
        # interval resolution is defined as how many points we will use to represent the posterior sample
        self.interval_resolution = interval_resolution # (R, )
        
        self.num_tasks = self.X_bounds.shape[0] # (N, )
        # grid search for points to evaluate
        self.X_distrib = Uniform(self.X_bounds[:, 0], self.X_bounds[:, 1])
        # self.X_grid = self.X_distrib.sample([interval_resolution]) # (R, N)
        self.X_grid = None
        
        # initializing our design matrix and target variable
        self.X = None # (D, N)
        self.y = None # (D, )

        # multitask Gaussian likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.initialize(noise=noise_level).to(dtype=dtype, device=device)
        self.gp = None
        self.dtype = dtype
        self.device = device

        self.minimum = torch.inf
        self.X_best = None

    def warm_start(self):
        self.X = torch.zeros((self.n_random_draws, self.num_tasks), 
                              dtype=self.dtype, device=self.device) # (n_random_draws, N)
        self.y = torch.zeros((self.n_random_draws,),
                                   dtype=self.dtype, device=self.device) # # (n_random_draws, )
        for i in range(self.n_random_draws):
            # sample = (self.X_bounds[:, 1] - self.X_bounds[:, 0]) * torch.rand(self.num_tasks) + self.X_bounds[:, 0] # (N, )
            sample = self.X_distrib.sample()
            observation = self.objective(sample)
            if observation < self.minimum:
                self.minimum = observation
                self.X_best = sample
            self.X[i] = sample
            self.y[i] = observation
        
    # process of choosing next point
    def choose_next_sample(self):
        
        # if we do not have enough samples, sample randomly from bounds
        if self.X is None:
            self.warm_start()
        
        # if we do, we fit the GP and choose the next point based on the posterior draw minimum
        else:
            # 1. Fit the GP to the observations we have
            if self.gp is None:
                # self.gp = MultitaskGPModel(self.X, self.y, self.num_tasks).to(dtype=self.dtype, device=self.device)
                self.gp = Matern_GP(self.X, self.y).to(dtype=self.dtype, device=self.device)
            else:
                self.gp.set_train_data(self.X, self.y, strict=False)
            
            # train the GP
            train_gp_hyperparams(self.gp, self.likelihood, self.X, self.y, lr=0.1, iters=50)
            
            # 2. Draw one sample (a function) from the posterior
            self.gp.eval()
            self.likelihood.eval()
            self.X_grid = self.X_distrib.sample([self.interval_resolution]) # (R, N)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_obs = self.likelihood(self.gp(self.X_grid))
                posterior_sample = pred_obs.sample() # (R, )

            # 3. Choose next point as the optimum of the sample
            which_min = torch.argmin(posterior_sample) # (1, )
            next_sample = self.X_grid[which_min] # (N, )
        
            # get the std from the posterior, for visualization purposes
            # posterior_mean = pred_obs.mean # (R, )
            # posterior_std = pred_obs.stddev # (R, )
        
            # let us observe the objective and append this new data to our X and y
            next_observation = self.objective(next_sample) # (1, )
            # TODO clip next_sample within bounds to ensure numerical stability
            if next_observation < self.minimum:
                self.minimum = next_observation
                self.X_best = next_sample
            self.X = torch.cat([self.X, next_sample[None, :]], dim=0)
            self.y = torch.cat([self.y, torch.atleast_1d(next_observation)], dim=0)
        
        return self.X, self.y
    
    def get_result(self):
        return self.X_best.detach().cpu().numpy(), self.minimum.item()