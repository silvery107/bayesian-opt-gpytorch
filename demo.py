# Ideas
# 1) Run evaluation for pushing task with parameters optimized by bayesian optimization
#       print confirmation info, print metrics, plot metrics, save to figures
# 2) Run eval with two baseline parametersn
#       print confirmation info, print metrics, plot metrics, save to figures
# * pushing task: which environment with which obstacle setup
# Note: print out an expected time to run before starting

import torch
import os
from optimizer.panda_pushing_optimizer import PandaBoxPushingStudy
from env.visualizers import GIFVisualizer
import numpy as np
import time
import random

seed = 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

def print_visualization_warning():
    print(TerminalColors.BOLD + TerminalColors.RED + "=====================================" + TerminalColors.ENDC)
    for _ in range(5):
        print(TerminalColors.BOLD + TerminalColors.RED + "Attention please: Visualization Window May Lay at the bottom!" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + TerminalColors.RED + "=====================================" + TerminalColors.ENDC)



if __name__ == "__main__":

    print(TerminalColors.BOLD + "=====================================" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "Here Is Demo From Yulun Zhuang & Ziqi Han. \n" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "We Provide Three Planar Pushing Scenes:\n" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "1) We Push Planar To Random Targets Without Obstacle.\n" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "2) We Push Planar With Obstacle In Easier Position.\n" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "3) We Push Planar With Obstacle In Harder Position.\n" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "Please Type Enter For Each Scenes To Start!\n" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "====================================" + TerminalColors.ENDC)


    EPOCH = 5
    RENDER = True
    LOGDIR = "logs/"

    if not os.path.exists(LOGDIR):
        os.mkdir(LOGDIR)

    # DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"
    visualizer = GIFVisualizer()
    
    print(TerminalColors.OKGREEN + f"Start Free pushing with random targets for {EPOCH} epoch(s)!" + TerminalColors.ENDC)
    print(TerminalColors.OKGREEN + f"Note that the visualization window may lay at the bottom!" + TerminalColors.ENDC)
    confirm_message = TerminalColors.OKGREEN + "Please Enter to continue..." + TerminalColors.ENDC
    confirm = input(confirm_message)
    print("Confirmed")
    print_visualization_warning()
    time.sleep(2)

    test_param_ours_obs = [0.05474454, 5.137514, 1.3101447, 8.374978]

    # visualizer.reset()
    test_free = PandaBoxPushingStudy(EPOCH, RENDER, LOGDIR, 
                                    study_name="test", 
                                    include_obstacle=False, 
                                    random_target=True,
                                    opt_type="test", 
                                    step_scale=0.1, 
                                    device=DEVICE,
                                    test_params=test_param_ours_obs,
                                    visualizer=visualizer)
    test_free.run()
    # visualizer.repeat_last_frame()
    # visualizer.get_gif("test_free.gif")
    

    print(TerminalColors.OKGREEN + f"Start Obstacle pushing Easy Mode for {EPOCH} epoch(s)!" + TerminalColors.ENDC)
    print(TerminalColors.OKGREEN + f"Note that the visualization window may lay at the bottom!" + TerminalColors.ENDC)
    confirm_message = TerminalColors.OKGREEN + "Please Enter to continue..." + TerminalColors.ENDC
    confirm = input(confirm_message)
    print("Confirmed")
    print_visualization_warning()
    time.sleep(2)

    # visualizer.reset()
    test_obs = PandaBoxPushingStudy(EPOCH, RENDER, LOGDIR, 
                                    study_name="test", 
                                    include_obstacle=True, 
                                    random_target=False,
                                    target_state=np.array([0.8, -0.1, 0]),
                                    opt_type="test", 
                                    step_scale=0.1, 
                                    device=DEVICE,
                                    test_params=test_param_ours_obs,
                                    visualizer=visualizer)
    test_obs.run()
    # visualizer.repeat_last_frame()
    # visualizer.get_gif("test_obs_easy.gif")


    print(TerminalColors.OKGREEN + f"Start Obstacle pushing Hard Mode for {EPOCH} epoch(s)!" + TerminalColors.ENDC)
    print(TerminalColors.OKGREEN + f"Note that the visualization window may lay at the bottom!" + TerminalColors.ENDC)
    confirm_message = TerminalColors.OKGREEN + "Please Enter to continue..." + TerminalColors.ENDC
    confirm = input(confirm_message)
    print("Confirmed")
    print_visualization_warning()
    time.sleep(2)

    # visualizer.reset()
    test_obs = PandaBoxPushingStudy(EPOCH, RENDER, LOGDIR, 
                                    study_name="test", 
                                    include_obstacle=True, 
                                    random_target=False,
                                    opt_type="test", 
                                    step_scale=0.1, 
                                    device=DEVICE,
                                    test_params=test_param_ours_obs,
                                    visualizer=visualizer)
    test_obs.run()
    # visualizer.repeat_last_frame()
    # visualizer.get_gif("test_obs_hard.gif")