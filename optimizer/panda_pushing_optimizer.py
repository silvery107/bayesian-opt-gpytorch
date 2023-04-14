import math
import os
import numpy as np
import torch
import csv
from time import time
import matplotlib.pyplot as plt

from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization as BayesOpt
from cma import CMAEvolutionStrategy

from controller.pushing_controller import PushingController, obstacle_avoidance_pushing_cost_function, free_pushing_cost_function
from env.panda_pushing_env import BOX_SIZE, PandaBoxPushingEnv
from optimizer.bayesian_optimization import BayesianOptimization
from model.state_dynamics_models import ResidualDynamicsModel

BOX_MULTI_RESIDUAL_MODEL = "assets/pretrained_models/box_multi_step_residual_dynamics_model.pt"

class PushingLogger:
    def __init__(self, study_name, opt_type, epoch) -> None:
        self.study_name = study_name
        self.opt_type = opt_type
        self.epoch = epoch
        self.reset()

    def reset(self):
        self.costs = []
        self.steps = []
        self.goal_dist = []
        self.goal_status = []
        self.iter_times = []
        self.params = []
        self.minimum = np.Inf
        # self.avg_costs = []
        # self.total_cost = 0.

    def update(self, cost, step, dist, status, param):
        # self.total_cost += cost
        self.costs.append(cost)

        # avg_cost = self.total_cost/len(self.costs)
        self.steps.append(step)
        self.goal_dist.append(dist)
        self.goal_status.append(status)
        if isinstance(param, list):
            self.params.append(param)
        else:
            self.params.append(param.tolist())
        print('-'*50)
        print(f"COST: {cost:.4f}")
        # print(f"AVG COST: {avg_cost:.4f}")
        print(f"STEP: {step}")
        print(f"GOAL: {status}")
        if cost < self.minimum:
            self.minimum = cost
            print(f"MIN COST: {self.minimum:.4f}")
            print(f"PARAM: {list(map('{:.4f}'.format, self.params[-1]))}")
    
    def to_ndarray(self):
        self.costs = np.asarray(self.costs)
        self.steps = np.asarray(self.steps)
        self.goal_dist = np.asarray(self.goal_dist)
        self.goal_status = np.asarray(self.goal_status)
        self.params = np.asarray(self.params)
        self.iter_times = np.asarray(self.iter_times)

    def update_time(self, iter_time):
        self.iter_times.append(iter_time)

    def save(self, log_dir):
        self.to_ndarray()
        print(f"Optimizer Average Suggest Time: {self.iter_times.mean()}")
        dataframe = np.stack([self.costs, self.steps, self.goal_dist, self.goal_status], axis=1)
        dataframe = np.concatenate((dataframe, self.params), axis=1)
        filename = f"{int(time())}_{self.study_name}_{self.opt_type}_{self.epoch}_{self.costs.min():.4f}.csv"
        save_path = os.path.join(log_dir, filename)
        with open(save_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(dataframe)

    def load(self, log_dir, filename):
        self.reset()
        load_path = os.path.join(log_dir, filename)
        with open(load_path, "r", newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                self.costs.append(row[0])
                self.steps.append(row[1])
                self.goal_dist.append(row[2])
                self.goal_status.append(row[3])
                self.params.append(row[4:])

    @property
    def length(self):
        return len(self.costs)

    def plot(self):
        # TODO plot results
        # plt.plot(np.arange(self.length), self.costs)
        # plt.show()
        # smoothed_costs = smooth(self.costs, 0.5)
        # plt.plot(np.arange(self.length), smoothed_costs)
        # plt.show()
        pass


class UnifiedBlackboxOptimizer:

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
            params = [self._optimizer.suggest()]
        
        elif self.opt_type == "cma":
            params = self._optimizer.ask()
        
        elif self.opt_type == "bayref":
            params = [list(self._optimizer.suggest(self._utility).values())]

        # return a list with at least one set of params for compatability
        self._params = params
        return params # list


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
            xval = self._optimizer.result[0] # ndarray
        
        elif self.opt_type == "bayref":
            xval = list(self._optimizer.max["params"].values()) # list

        return xval

class PandaBoxPushingStudy:
    def __init__(self, epoch, render, logdir, study_name, 
                 include_obstacle=False, 
                 random_target=False, target_state=None, 
                 opt_type="bayes", device="cpu", 
                 step_scale=0.1, goal_scale=10.):
        # TODO set obstacle pose?
        self._epoch = epoch
        self._n_step = 20
        self._render = render
        self._log_dir = logdir
        self._random_target = True if not include_obstacle and random_target else False
        self._target_state = target_state
        self._opt_type = opt_type

        param_dict = {}
        param_dict["lower"] = [0, 0, 0, 0]
        param_dict["upper"] = [1, 10, 10, 10]
        param_dict["acq_mode"] = "ei"
        param_dict["initial_mean"] = [0.5, 5, 5, 5]
        param_dict["initial_sigma"] = 0.5
        param_dict["popsize"] = 3

        cost_func = obstacle_avoidance_pushing_cost_function if include_obstacle else free_pushing_cost_function

        self._optimizer = UnifiedBlackboxOptimizer(opt_type, param_dict, device)
        self._target_state = np.array([0.7, 0., 0.])

        self._step_scale = step_scale
        self._goal_scale = goal_scale
        self._goal_tol = BOX_SIZE

        self._dynamics_model = ResidualDynamicsModel(state_dim=3, action_dim=3)
        self._dynamics_model.load_state_dict(torch.load(BOX_MULTI_RESIDUAL_MODEL))
        self._dynamics_model.eval()

        self._env = PandaBoxPushingEnv(debug=render, include_obstacle=include_obstacle, render_non_push_motions=False, 
                                 camera_heigh=800, camera_width=800, render_every_n_steps=5)
        self._controller = PushingController(self._env, self._dynamics_model,
                                            cost_func, 
                                            num_samples=1000, horizon=20,
                                            device=device)
        suffix = "_obstacle" if include_obstacle else "_free"
        self._logger = PushingLogger(study_name+suffix, opt_type, epoch)

        if not os.path.exists(logdir):
            os.mkdir(logdir)

    def _compute_cost(self, goal_distance, goal_reached, n_step):
        cost = goal_distance + self._step_scale * n_step + (not goal_reached) * self._goal_scale
        return cost

    def run(self):
        self._logger.reset()
        if self._target_state is not None:
            target_state = self._target_state.copy()
        else:
            target_state = get_random_target_state()
        opt_epoch = self._epoch // 3 if self._opt_type=="cma" else self._epoch
        print(f"Optimizing box pushing using {self._opt_type.upper()} optimizer for {self._epoch} epoches")
        for _ in range(opt_epoch):
            start_time = time()
            parameters = self._optimizer.suggest()
            suggest_time = time() - start_time
            self._logger.update_time(suggest_time)
            # Run trial
            for param in parameters:
                if self._random_target:
                    target_state = get_random_target_state()
                end_state, pushing_step = run_pushing_task(self._env, self._controller, self._n_step, target_state, param)
                goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
                goal_reached = goal_distance < self._goal_tol
                cost = self._compute_cost(goal_distance, goal_reached, pushing_step)

                self._logger.update(cost, pushing_step, goal_distance, goal_reached, param)
                self._optimizer.register(param, cost)

        # optimized_param = self._optimizer.get_result()
        self._optimizer.print_result()

        self._logger.save(self._log_dir)
        if self._render:
            self._env.disconnect()

    def plot_results(self):
        self._logger.plot()

    def load_logs(self, filename):
        self._logger.load(self._log_dir, filename)

def get_random_target_state(low=[0.65, -0.3, 0.0], high=[0.8, 0.3, 0.0]):
    return np.random.uniform(low=low, high=high, size=None)

def run_pushing_task(env, controller, step, target_state, hyperparameters):
    # init env
    env.set_target_state(target_state)
    state = env.reset()
    # init controller
    controller.reset()
    controller.set_target_state(target_state) #* set target_state before set params
    controller.set_parameters(hyperparameters)
    # for i in tqdm(range(step)):
    for i in range(step):
        action = controller.control(state)
        state, _, done, _ = env.step(action)
        if done:
            break
    end_state = env.get_state()
    return end_state, i

def smooth(scalars: list, weight: float) -> list:
    """
    Exponential moving average (EMA) implementation according to tensorboard
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        # Calculate smoothed value
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed