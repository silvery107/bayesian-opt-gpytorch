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
import cma
import argparse
from model.state_dynamics_models import ResidualDynamicsModel
from controller.pushing_controller import PushingController, obstacle_avoidance_pushing_cost_function, free_pushing_cost_function
from optimizer.bayesian_optimization import BayesianOptimization
from env.panda_pushing_env import PandaBoxPushingEnv, TARGET_POSE_OBSTACLES_BOX, BOX_SIZE, TARGET_POSE_FREE_BOX

def get_total_cost(end_state, target_state, n_steps, k = 0.1):

    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance > BOX_SIZE
    cost = goal_distance + k * (n_steps - 5) + goal_reached * 10
    return cost


def target_state_reset():
    
    return np.random.uniform(low=[0.5, -0.35, 0.0], high=[0.8, 0.35, 0.0], size=None)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="bayes", choices=["bayes", "cma", "baynb", "eval"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--cuda", action="store_false")

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"

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
    counter = 0
    # target_state = TARGET_POSE_OBSTACLES_BOX
    target_state = TARGET_POSE_OBSTACLES_BOX

    if args.optimizer == "cma":
        # cma test example
        # initial_mean = [0, 0, 0, 0]
        initial_mean = [3.583136148286834, 2.7933472511972566, 3.0060063190645026, 1.947791653714797]
      
        initial_sigma = 0.5
        popsize = 2 # 2
        optimizer = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, {'bounds': [0, 10], 'popsize': popsize})
        
        cost_list = []
        goal_reached_times = 0
        total_trail_times = 0
        cost_total = 0

        for _ in range(args.epoch):

            parameters = optimizer.ask(number=popsize)
            fit = []
            for hyperparameter in parameters:
                controller.set_parameters(hyperparameter)
                start_state = env.reset()
                state = start_state
                for i in tqdm(range(args.step)):
                    # action = None
                    # for hyperparameter in parameters:
                        # controller.set_parameters(hyperparameter)
                    action = controller.control(state)
                    state, reward, done, _ = env.step(action)
                    if done:
                        break

                controller.mppi.reset()
                end_state = env.get_state()
                target_state = TARGET_POSE_OBSTACLES_BOX
                goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
                goal_reached = goal_distance < BOX_SIZE
                total_trail_times += 1
                if goal_reached:
                    goal_reached_times += 1
                cost = get_total_cost(end_state, target_state, i)
                cost_total += cost
                ave_cost = cost_total / (len(cost_list) + 1)
                cost_list.append(ave_cost)
                fit.append(cost)
                print(f'GOAL REACHED: {goal_reached}')
                print(f'COST : {cost}')
                print(f'AVERAGE COST : {ave_cost}')
        
            optimizer.tell(parameters, fit)
            optimizer.result_pretty()
            # Evaluate if goal is reached
            
        print(f"Reach Goal Times {goal_reached_times}")
        print(f"Total Goal Times {total_trail_times}")

        plt.plot(np.arange(0,len(cost_list),1),cost_list)
        plt.show()
        optimizer.result_pretty()

    elif args.optimizer == "bayes":
        optimizer = BayesianOptimization(torch.tensor([0, 0, 0, 0]), torch.tensor([1, 10, 10, 10]), acq_mode='ts', device=device)
        cost_list = []
        goal_reached_times = 0
        cost_total = 0
        for _ in range(args.epoch):
            # target_state = target_state_reset()
            start_state = env.reset()
            state = start_state

            parameters = optimizer.suggest()
            controller.set_parameters(parameters)
            # env.target_state = target_state
            # controller.set_target_state(target_state)
            # print(parameters)
            # controller.set_parameters([2.5275327272262276, 2.5476387164342835, 0.3065728561574313, 2.7788383937981758])
            

            for i in tqdm(range(args.step)):
                # parameters = optimizer.suggest()
                action = controller.control(state)
                # cost_min = controller.get_cost_total().min()
                state, reward, done, _ = env.step(action)
                if done:
                    break
            
            controller.reset()
            # Evaluate if goal is reached
            end_state = env.get_state()
            goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
            goal_reached = goal_distance < BOX_SIZE
            if goal_reached:
                goal_reached_times += 1
            cost = get_total_cost(end_state, target_state, i)

            cost_total += cost
            ave_cost = cost_total / (len(cost_list) + 1)
            cost_list.append(ave_cost)

            optimizer.register(torch.tensor(cost))
            print(f'COST : {cost}')
            print(f'AVERAGE COST : {ave_cost}')
            print(f'GOAL REACHED: {goal_reached}')
        
        xval, fval = optimizer.get_result()
        
        print(f"Found minimum objective {fval:.4f} at {xval}")
        print(f"Reach Goal Times {goal_reached_times}")
        plt.plot(np.arange(0,len(cost_list),1),cost_list)
        plt.show()
        
    elif args.optimizer == "baynb":
        from bayes_opt import BayesianOptimization, UtilityFunction
        optimizer = BayesianOptimization(f=None, pbounds={"lambda": (0., 1), 
                                                          "sigma1": (1e-7, 10.),
                                                          "sigma2": (1e-7, 10.),
                                                          "sigma3": (1e-7, 10.)})
        utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
        cost_list = []
        cost_avg_list = []
        cost_total = 0.
        for _ in range(args.epoch):
            # target_state = target_state_reset()
            start_state = env.reset()
            state = start_state

            params = optimizer.suggest(utility)
            parameters = list(params.values())
            # print(parameters)
            controller.set_parameters(parameters)
            # env.target_state = target_state
            # controller.set_target_state(target_state)

            for i in tqdm(range(args.step)):
                action = controller.control(state)
                state, reward, done, _ = env.step(action)
                if done:
                    break
            
            controller.reset()
            # Evaluate if goal is reached
            end_state = env.get_state()
            goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
            goal_reached = goal_distance < BOX_SIZE

            cost = get_total_cost(end_state, target_state, i)
            optimizer.register(params=params, target=cost)

            cost_total += cost
            cost_avg = cost_total/(len(cost_list)+1)
            cost_list.append(cost)
            cost_avg_list.append(cost_avg)
            print(f"AVG COST: {cost_avg}")
            print(f'COST: {cost}')
            print(f'GOAL REACHED: {goal_reached}')
        
        plt.plot(np.arange(len(cost_list)), cost_list)
        plt.show()
        plt.plot(np.arange(len(cost_avg_list)), cost_avg_list)
        plt.show()
        print(optimizer.max)
    else:
        # TODO evaluation
        pass

    if args.render:
        env.disconnect()