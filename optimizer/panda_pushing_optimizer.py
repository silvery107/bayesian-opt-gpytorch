import torch


from optimizer.bayesian_optimization import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization as BayesOpt
from cma import CMAEvolutionStrategy


class PandaPushingOptimizer:

    def __init__(self, opt_type:str, param_dict:dict={}, device="cpu") -> None:
        self.opt_type = opt_type
        self.device = device

        lower = param_dict.get("lower", [0, 0, 0, 0])
        upper = param_dict.get("upper", [1, 10, 10, 10])
        if opt_type == "bayes":
            acq_mode = param_dict.get("acq_mode", "ei")
            self._optimizer = BayesianOptimization(torch.tensor(lower),
                                                   torch.tensor(upper), 
                                                   acq_mode=acq_mode, 
                                                   device=device)
            
        elif opt_type == "cma":
            initial_mean = param_dict.get("initial_mean", [0, 0, 0, 0])
            # initial_mean = [3.583136148286834, 2.7933472511972566, 3.0060063190645026, 1.947791653714797]
            initial_sigma = param_dict.get("initial_sigma", 0.5)
            popsize = param_dict.get("popsize", 2)
            bounds = [min(lower), max(upper)]
            self._optimizer = CMAEvolutionStrategy(initial_mean, initial_sigma, {'bounds': bounds, 'popsize': popsize})
            self._popsize = popsize
            self._targets = []
        
        elif opt_type == "bayref":
            bounds = {f"param_{i}": (l, u) for i, (l, u) in enumerate(zip(lower, upper))}
            # bounds={"lambda": (0., 1), "sigma1": (1e-7, 10.), "sigma2": (1e-7, 10.), "sigma3": (1e-7, 10.)}
            self._optimizer = BayesOpt(f=None, pbounds=bounds)
            self._utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

        self._params = None
        

    def suggest(self):
        if self.opt_type == "bayes":
            params = [self._optimizer.suggest()] # tensor list
        
        elif self.opt_type == "cma":
            params = self._optimizer.ask() # list
        
        elif self.opt_type == "bayref":
            params = [list(self._optimizer.suggest(self._utility).values())] # list

        # return a list with at least one set of params for compatability
        self._params = params
        return params


    def register(self, params, target):
        if self.opt_type == "bayes":
            self._optimizer.register(torch.tensor(target, device=self.device))
        
        elif self.opt_type == "cma":
            if len(self._targets) < self._popsize:
                self._targets.append(target)
            else:
                self._optimizer.tell(self._params, self._targets)
                self._targets = []

        elif self.opt_type == "bayref":
            self._optimizer.register(params=params, target=target)

    def print_result(self):
        if self.opt_type == "bayes":
            xval, fval = self._optimizer.get_result()
            print(f"Found minimum objective {fval:.4f} at {xval}")

        elif self.opt_type == "cma":
            self._optimizer.result_pretty()
        
        elif self.opt_type == "bayref":
            print("Target:", self._optimizer.max["target"])
            print("Params:", self._optimizer.max["params"])

    def get_result(self):
        if self.opt_type == "bayes":
            xval, fval = self._optimizer.get_result() # ndarray

        elif self.opt_type == "cma":
            xval = self._optimizer.result[0] # list
        
        elif self.opt_type == "bayref":
            xval = list(self._optimizer.max["params"].values()) # list

        return xval