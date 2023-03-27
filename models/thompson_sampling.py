import torch
import gpytorch
from models.gp_models import Matern_GP, train_gp_hyperparams


# TS-GP optimizer
class ThompsonSamplingGP:
    
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
                self.gp = Matern_GP(self.X, self.y)
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
        
            # let us also get the std from the posterior, for visualization purposes
            posterior_mean = pred_obs.mean
            posterior_std = pred_obs.stddev
        
            # let us observe the objective and append this new data to our X and y
            next_observation = self.objective(next_sample)
            self.X = torch.cat([self.X, torch.atleast_1d(next_sample)])
            self.y = torch.cat([self.y, torch.atleast_1d(next_observation)])
        
        # return everything if possible
        try:
            # returning values of interest
            return self.X, self.y, self.X_grid, posterior_sample, posterior_mean, posterior_std
        
        # if not, return whats possible to return
        except:
            return (self.X, self.y, self.X_grid, torch.tensor([torch.mean(self.y)]*self.interval_resolution), 
                    torch.tensor([torch.mean(self.y)]*self.interval_resolution), torch.tensor([0]*self.interval_resolution))