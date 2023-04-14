import torch
import os

import argparse
from optimizer.panda_pushing_optimizer import PandaBoxPushingStudy

import argparse

# Data to collect for each optimizer:
# Cost, n_step, goal_reach, goal_dist, opt_iter_time

# Reward to analyze: k = [0.05, 0.1, 0.2, 0.4]

# Env to evaluate: [box, disk]

# Obstacles to evaluate: 
# 1) free pushing with random target state 
# 2) pushing against a non-trivial set of obstacles

# Model to evaluate: multi_step_residual_model

# Visual results: metrics TABLE, training curve PLOT, evaluation curve PLOT


# BOX/DISK -- FREE/OBSTACLS -- COST_FACTOR -- OPTIMIZER

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--logdir", type=str, default="logs")

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    device = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    
    # BOX -- FREE -- k=0.1 -- bayes
    study1 = PandaBoxPushingStudy(args.epoch, args.render, args.logdir, 
                                  study_name="study1", 
                                  include_obstacle=False, 
                                  random_target=True,
                                  opt_type="bayes", 
                                  step_scale=0.1, 
                                  device=device)
    study1.run()
    study1.plot_results()


    # BOX -- OBSTACLE -- k=0.1 -- bayes
    study2 = PandaBoxPushingStudy(args.epoch, args.render, args.logdir, 
                                  study_name="study2", 
                                  include_obstacle=True, 
                                  random_target=False,
                                  opt_type="bayes", 
                                  step_scale=0.1, 
                                  device=device)
    study2.run()
    study2.plot_results()