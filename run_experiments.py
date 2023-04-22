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
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--test", action="store_true", default=False)


    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    device = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    
    if args.test:
        args.epoch = 100
        args.logdir = "logs/handmade"
        test_param1 = [0.5,2,2,1] #zhuang
        # test_param2 = [0.05474454,5.137514,1.3101447,8.374978] #bayes1000
        # test_param3 = [0.21833461408187976, 5.106353269081676, 3.660409505665367, 5.96104154751906]#cma500
        
        test_param4 = [0.5,2,2,1]
        # test_param5 = [0.05474454,5.137514,1.3101447,8.374978] #bayes1000
        # test_param6 = [0.21833461408187976, 5.106353269081676, 3.660409505665367, 5.96104154751906]#cma500
        
        test_params_obs = []
        test_params_free = []
        test_params_obs.append(test_param1)
        # test_params_obs.append(test_param2)
        # test_params_obs.append(test_param3)

        test_params_free.append(test_param4)
        # test_params_free.append(test_param5)
        # test_params_free.append(test_param6)

        for test_param in test_params_obs:
            test = PandaBoxPushingStudy(args.epoch, args.render, args.logdir, 
                                            study_name="test_obs", 
                                            include_obstacle=True, 
                                            random_target=False,
                                            opt_type="test", 
                                            step_scale=0.1, 
                                            device=device,
                                            test_params=test_param)
            test.run()
            test.plot_results()
        
        for test_param in test_params_free:
            test = PandaBoxPushingStudy(args.epoch, args.render, args.logdir, 
                                            study_name="test_free", 
                                            include_obstacle=False, 
                                            random_target=True,
                                            opt_type="test", 
                                            step_scale=0.1, 
                                            device=device,
                                            test_params=test_param)
            test.run()
            test.plot_results()

    else:
        args.logdir = "logs/epoch_1000"

        # BOX -- OBSTACLE -- k=0.1 -- bayes
        study1 = PandaBoxPushingStudy(args.epoch, args.render, args.logdir, 
                                    study_name="study1", 
                                    include_obstacle=True, 
                                    random_target=False,
                                    opt_type="bayref", 
                                    step_scale=0.1, 
                                    device=device)
        study1.run()
        study1.plot_results()

        # study2 = PandaBoxPushingStudy(args.epoch, args.render, args.logdir, 
        #                             study_name="study2", 
        #                             include_obstacle=True, 
        #                             random_target=False,
        #                             opt_type="bayref", 
        #                             step_scale=0.1, 
        #                             device=device)
        # study2.run()
        # study2.plot_results()

        # study3 = PandaBoxPushingStudy(args.epoch, args.render, args.logdir, 
        #                             study_name="study3", 
        #                             include_obstacle=True, 
        #                             random_target=False,
        #                             opt_type="cma", 
        #                             step_scale=0.1, 
        #                             device=device)
        # study3.run()
        # study3.plot_results()

        # study4 = PandaBoxPushingStudy(args.epoch, args.render, args.logdir, 
        #                             study_name="study4", 
        #                             include_obstacle=False, 
        #                             random_target=True,
        #                             opt_type="bayes", 
        #                             step_scale=0.1, 
        #                             device=device)
        # study4.run()
        # study4.plot_results()


        # study5 = PandaBoxPushingStudy(args.epoch, args.render, args.logdir, 
        #                             study_name="study5", 
        #                             include_obstacle=False, 
        #                             random_target=True,
        #                             opt_type="bayref", 
        #                             step_scale=0.1, 
        #                             device=device)
        # study5.run()
        # study5.plot_results()


        # study6 = PandaBoxPushingStudy(args.epoch, args.render, args.logdir, 
        #                             study_name="study6", 
        #                             include_obstacle=False, 
        #                             random_target=True,
        #                             opt_type="cma", 
        #                             step_scale=0.1, 
        #                             device=device)
        # study6.run()
        # study6.plot_results()
