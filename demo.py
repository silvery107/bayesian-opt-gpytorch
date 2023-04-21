# Ideas
# 1) Run evaluation for pushing task with parameters optimized by bayesian optimization
#       print confirmation info, print metrics, plot metrics, save to figures
# 2) Run eval with two baseline parametersn
#       print confirmation info, print metrics, plot metrics, save to figures
# * pushing task: which environment with which obstacle setup
# Note: print out an expected time to run before starting

import torch
import os
import argparse
from optimizer.panda_pushing_optimizer import PandaBoxPushingStudy
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--test", action="store_true", default=True)

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    device = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    
    if args.test:
        args.epoch = 10
        args.logdir = "logs/test/free"

        test_param_obs = []
        test_param_ours_obs = [0.12118153, 0.768279705, 0.332457308, 7.487837782]
        test_param_cma_obs = [0.12118153, 0.768279705, 0.332457308, 7.487837782]
        test_param_bayref_obs = [0.12118153, 0.768279705, 0.332457308, 7.487837782]
        test_param_obs.append(test_param_ours_obs)
        test_param_obs.append(test_param_cma_obs)
        test_param_obs.append(test_param_bayref_obs)


        for i in test_param_obs:
            test_obs = PandaBoxPushingStudy(args.epoch, args.render, args.logdir, 
                                            study_name="test", 
                                            include_obstacle=True, 
                                            random_target=False,
                                            opt_type="test", 
                                            step_scale=0.1, 
                                            device=device,
                                            test_params=i)
            test_obs.run()
            test_obs.plot_results()

        test_param_free = []
        test_param_ours_free = [0.12118153, 0.768279705, 0.332457308, 7.487837782]
        test_param_cma_free = [0.12118153, 0.768279705, 0.332457308, 7.487837782]
        test_param_bayref_free = [0.12118153, 0.768279705, 0.332457308, 7.487837782]
        test_param_free.append(test_param_ours_free)
        test_param_free.append(test_param_cma_free)
        test_param_free.append(test_param_bayref_free)

        for i in test_param_free:
            test_free = PandaBoxPushingStudy(args.epoch, args.render, args.logdir, 
                                            study_name="test", 
                                            include_obstacle=False, 
                                            random_target=True,
                                            opt_type="test", 
                                            step_scale=0.1, 
                                            device=device,
                                            test_params=i)
            test_free.run()
            test_free.plot_results()

    