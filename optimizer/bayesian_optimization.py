import torch
from torch import Tensor
import numpy as np
import random
import gpytorch

from model.gp_models import Matern_GP, train_gp_hyperparams
from torch.distributions.uniform import Uniform
from torch.distributions import Normal


class AcquisitionFunction:
    def __init__(self, mode="ts") -> None:
        """
        mode: {"ts", "ei"}
            * "ts" is the Thompson Sampling method
            * "ei" is the Expected Improvement method
        """
        if mode not in ["ts", "ei"]:
            err = "The acquisition function " \
                  "{} has not been implemented, " \
                  "please choose one of ts or ei.".format(mode)
            raise NotImplementedError(err)
        self.mode = mode
        self.norm = Normal(0., 1.)
    
    def apply(self, pred_distrib, y_min):
        if self.mode == "ts":
            return self._thompson_sampling(pred_distrib)
        else:
            return self._expected_improvement(pred_distrib, y_min)

    def _thompson_sampling(self, distrib):
        return distrib.sample()

    def _expected_improvement(self, distrib, y_min):
        a = (distrib.mean + y_min)
        z = a / (distrib.stddev)
        return a * self.norm.cdf(z) + distrib.stddev * self.norm.log_prob(z).exp()


class BayesianOptimization:
    def __init__(self, lower_bounds:Tensor, upper_bounds:Tensor, acq_mode:str="ei", 
                 seed:int=42, n_warmup:int=5, n_grid_pt:int=10000, dtype=torch.float32, device="cpu"):
        self.model = None
        self.likelihood = None
        self.acq = AcquisitionFunction(mode=acq_mode)
        assert lower_bounds.shape == upper_bounds.shape
        if not lower_bounds.shape:
            self.X_bounds = torch.stack([lower_bounds.unsqueeze(0), 
                                         upper_bounds.unsqueeze(0)], 
                                         dim=1).to(dtype=dtype, device=device) # (1, 2)
        else:
            self.X_bounds = torch.stack([lower_bounds, 
                                         upper_bounds], 
                                         dim=1).to(dtype=dtype, device=device) # (N, 2)
        # print(self.X_bounds)
        self.X_distrib = Uniform(self.X_bounds[:, 0], self.X_bounds[:, 1])


        self.dtype = dtype
        self.device = device
        self.num_tasks = self.X_bounds.shape[0] # N
        self.n_warmup = n_warmup
        assert self.n_warmup > 0
        self.n_grid_pt = n_grid_pt # R

        
        self._minimum = torch.inf
        self._X_best = None
        self._X_last = None
        self._X = torch.tensor([], dtype=dtype, device=device) # (D, N)
        self._y = torch.tensor([], dtype=dtype, device=device) # (D, )

        self._noise_level = 1e-3
        self._model_lr = 0.1
        self._model_epoch = 50
        # self.reset_seed(seed)
    
    def minimize(self, objective, n_iter:int):
        assert n_iter > self.n_warmup
        for _ in range(n_iter):
            next_sample = self.suggest()
            next_objective = objective(next_sample)
            self.register(next_objective)
            
        xval, fval = self.get_result()
        return xval, fval

    def suggest(self) -> Tensor:
        if self._X.shape[0] <= self.n_warmup:
            next_sample = self._sample_warmup()
        else:
            # 1. Fit the GP to the observations we have
            self._fit_model(self._X, self._y)
            # 2. Draw one sample (a function) from the surrogate function3
            X_grid = self.X_distrib.sample([self.n_grid_pt]) # (R, N)
            pred_obs_distrib = self._surrogate(X_grid)
            # 3. Choose next point as the optimum of the sample
            pred_obs_func = self.acq.apply(pred_obs_distrib, self._minimum) # (R, )
            which_min = torch.argmin(pred_obs_func) # (1, )
            
            next_sample = X_grid[which_min] # (N, )
        
        self._X_last = next_sample
        return next_sample.clamp(self.X_bounds[:, 0], self.X_bounds[:, 1])

    def register(self, y_new:Tensor):
        self._X = torch.cat([self._X, self._X_last[None, :]], dim=0)
        self._y = torch.cat([self._y, torch.atleast_1d(y_new).to(dtype=self.dtype, device=self.device)], dim=0)
        if y_new < self._minimum:
            # print(f"Got new minimum {y_new}, update X_best with {self._X_last}")
            self._minimum = y_new
            self._X_best = self._X_last

    def _sample_warmup(self):
        # sample = (self.X_bounds[:, 1] - self.X_bounds[:, 0]) * torch.rand(self.num_tasks) + self.X_bounds[:, 0]
        sample = self.X_distrib.sample() # (N, )
        return sample

    def _fit_model(self, X:Tensor, y:Tensor):
        if self.model is None:
            self.model = Matern_GP(X, y).to(dtype=self.dtype, device=self.device)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.likelihood.initialize(noise=self._noise_level).to(dtype=self.dtype, device=self.device)
        else:
            self.model.set_train_data(X, y, strict=False)
        # Train the GP
        train_gp_hyperparams(self.model, self.likelihood, self._X, self._y, lr=self._model_lr, iters=self._model_epoch)

    def _surrogate(self, X:Tensor)->Tensor:
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_obs_distrib = self.likelihood(self.model(X))
            # posterior_sample = pred_obs.sample() # (R, )
        
        return pred_obs_distrib

    def get_result(self):
        return self._X_best.detach().cpu().numpy(), self._minimum.item()
    
    def get_dataset(self):
        return self._X, self._y

    def reset_seed(self, seed):
        """
        Reset random seed to the specific number
        Inputs:
        - number: A seed number to use
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return