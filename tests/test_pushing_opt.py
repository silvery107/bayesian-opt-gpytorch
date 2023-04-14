# working directory
import os
import inspect
from matplotlib import pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import torch
from tqdm import tqdm
import numpy as np
import argparse

from model.state_dynamics_models import ResidualDynamicsModel
from controller.pushing_controller import PushingController, obstacle_avoidance_pushing_cost_function, free_pushing_cost_function
from env.panda_pushing_env import PandaBoxPushingEnv, TARGET_POSE_OBSTACLES_BOX, BOX_SIZE, TARGET_POSE_FREE_BOX
from optimizer.panda_pushing_optimizer import UnifiedBlackboxOptimizer

def get_pushing_res(end_state, target_state, n_step, k=0.1, tol=0.1):
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < tol
    cost = goal_distance + k * (n_step - 5) + (not goal_reached) * 10
    return cost, goal_reached

def get_random_target_state(low=[0.5, -0.35, 0.0], high=[0.8, 0.35, 0.0]):
    return np.random.uniform(low=low, high=high, size=None)

def run_box_pushing_objective(env, controller, step, target_state, hyperparameters):
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
    return get_pushing_res(end_state, target_state, i, tol=BOX_SIZE)

def eval_panda_pushing(args, env, controller, parameters=None):
    if parameters is None:
        parameters = controller.default_parameters

    cost_list = []
    cost_avg_list = []
    cost_total = 0.
    goal_reached_times = 0
    for _ in range(args.epoch):
        target_state = get_random_target_state()
        print('-'*100)
        cost, goal_reached = run_box_pushing_objective(env, controller, args.step, target_state, parameters)
        if goal_reached:
            goal_reached_times += 1
        cost_total += cost
        cost_avg = cost_total/(len(cost_list)+1)
        cost_list.append(cost)
        cost_avg_list.append(cost_avg)
        print(f"AVG COST: {cost_avg}")
        print(f'COST: {cost}')
        print(f'GOAL REACHED: {goal_reached}')
    
    print(f"Reach Goal Times {goal_reached_times}")
    plt.plot(np.arange(len(cost_list)), cost_list)
    plt.show()
    plt.plot(np.arange(len(cost_avg_list)), cost_avg_list)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="bayes", choices=["bayes", "cma", "bayref"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"

    # load pre-trained box pushing model
    load_path = 'assets/pretrained_models/box_multi_step_residual_dynamics_model.pt'
    state_dict = torch.load(load_path)
    box_multi_step_residual_dynamics_model = ResidualDynamicsModel(state_dim=3, action_dim=3)
    box_multi_step_residual_dynamics_model.load_state_dict(state_dict)
    box_multi_step_residual_dynamics_model.eval()

    env = PandaBoxPushingEnv(debug=args.render, include_obstacle=True, render_non_push_motions=False, 
                             camera_heigh=800, camera_width=800, render_every_n_steps=5)
    controller = PushingController(env, box_multi_step_residual_dynamics_model,
                                   obstacle_avoidance_pushing_cost_function, 
                                   num_samples=1000, horizon=20,
                                   device=device)
    
    # target_state = TARGET_POSE_OBSTACLES_BOX
    _target_state = TARGET_POSE_OBSTACLES_BOX

    optimizer = UnifiedBlackboxOptimizer(args.optimizer, device=device)
    
    cost_list = []
    cost_avg_list = []
    cost_total = 0.
    goal_reached_times = 0
    total_trail_times = 0
    opt_epoch = args.epoch
    if args.epoch < 3 and args.optimizer=="cma":
        print(f"Epoch number {args.epoch} is too small for {args.optimizer.upper()} optimizer, set to 3 instead.")
        opt_epoch = 3
    print(f"Optimizing box pushing using {args.optimizer.upper()} optimizer for {opt_epoch} epoches with {args.step} steps")
    for _ in range(opt_epoch):
        parameters = optimizer.suggest()
        for param in parameters:
            print('-'*50)
            # target_state = target_state_reset()
            cost, goal_reached = run_box_pushing_objective(env, controller, args.step, _target_state, param)
            optimizer.register(param, cost)

            total_trail_times += 1
            if goal_reached:
                goal_reached_times += 1
            cost_total += cost
            cost_avg = cost_total/(len(cost_list)+1)
            cost_list.append(cost)
            cost_avg_list.append(cost_avg)
            print(f'COST: {cost}')
            print(f"AVG COST: {cost_avg}")
            print(f'GOAL REACHED: {goal_reached}')

    optimizer.print_result()
    print(f"Reach Goal Times {goal_reached_times} / {total_trail_times}")
    plt.plot(np.arange(len(cost_list)), cost_list)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()
    plt.plot(np.arange(len(cost_avg_list)), cost_avg_list)
    plt.xlabel("Epoch")
    plt.ylabel("Average Cost")
    plt.show()

    optimized_param = optimizer.get_result()
    print(f"Evaluating box pushing with params {optimized_param} for {args.epoch} epoches with {args.step} steps")
    eval_panda_pushing(args, env, controller, optimized_param)

    if args.render:
        env.disconnect()