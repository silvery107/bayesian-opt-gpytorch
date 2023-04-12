# working directory
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import torch
from tqdm import tqdm
import numpy as np
import cma
import argparse

from model.state_dynamics_models import ResidualDynamicsModel
from controller.pushing_controller import PushingController, obstacle_avoidance_pushing_cost_function
from optimizer.bayesian_optimization import BayesianOptimization
from env.panda_pushing_env import PandaBoxPushingEnv, BOX_SIZE

def get_total_cost(end_state, target_state, n_steps, k = 0.1, n_collision = 0):

    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance > BOX_SIZE
    cost = goal_distance + k * n_steps + goal_reached * 10
    return cost


def target_state_reset():
    
    return np.random.uniform(low=[0.05, -0.35], high=[.8, 0.35], size=None)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="bayes", choices=["bayes", "cma"])
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load pre-trained box pushing model
    load_path = 'assets/pretrained_models/box_multi_step_residual_dynamics_model.pt'
    state_dict = torch.load(load_path)
    box_multi_step_residual_dynamics_model = ResidualDynamicsModel(state_dim=3, action_dim=3)
    box_multi_step_residual_dynamics_model.load_state_dict(state_dict)
    box_multi_step_residual_dynamics_model.eval()

    env = PandaBoxPushingEnv(debug=args.render, include_obstacle=True, render_non_push_motions=False, camera_heigh=800, camera_width=800, render_every_n_steps=5)
    controller = PushingController(env, box_multi_step_residual_dynamics_model,
                                   obstacle_avoidance_pushing_cost_function, 
                                   num_samples=1000, horizon=20,
                                   device=device)
    env.reset()
    optimizer = None

    if args.optimizer == "cma":
        # cma test example
        initial_mean = [0, 0, 0, 0]
        initial_sigma = 0.5
        popsize = 8 # 2
        optimizer = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, {'bounds': [0, 1], 'popsize': popsize})

        for _ in range(args.epoch):
            start_state = env.reset()
            state = start_state

            for i in tqdm(range(args.step)):
                parameters = optimizer.ask(number=popsize)
                fit = []
                action = None
                for hyperparameter in parameters:
                    controller.set_parameters(hyperparameter)
                    action = controller.control(state)
                    fit.append(controller.get_cost_total().min())
                optimizer.tell(parameters, fit)
                state, reward, done, _ = env.step(action)
                if done:
                    break
            
            controller.mppi.reset()
            # Evaluate if goal is reached
            end_state = env.get_state()
            target_state = env.target_state
            goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
            goal_reached = goal_distance < BOX_SIZE
            print(f'GOAL REACHED: {goal_reached}')
            
        optimizer.result_pretty()

    elif args.optimizer == "bayes":
        optimizer = BayesianOptimization(torch.tensor([0, 0, 0, 0]), torch.tensor([1, 1, 1, 1]), device=device)

        for _ in range(args.epoch):
            start_state = env.reset()
            env.target_state = target_state_reset()
            print(env.target_state)
            state = start_state

            parameters = optimizer.suggest()
            controller.set_parameters(parameters)

            for i in tqdm(range(args.step)):
                # parameters = optimizer.suggest()
                action = controller.control(state)
                # cost_min = controller.get_cost_total().min()
                state, reward, done, _ = env.step(action)
                if done:
                    break
            
            controller.mppi.reset()
            # Evaluate if goal is reached
            end_state = env.get_state()
            target_state = env.target_state
            goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
            goal_reached = goal_distance < BOX_SIZE

            cost = get_total_cost(end_state, target_state, i)
            optimizer.register(torch.tensor(cost))
            print(f'COST : {cost}')
            print(f'GOAL REACHED: {goal_reached}')
        
        xval, fval = optimizer.get_result()
        print(f"Found minimum objective {fval:.4f} at {xval}")

    if args.render:
        env.disconnect()