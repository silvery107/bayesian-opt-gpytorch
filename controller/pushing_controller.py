import torch
from controller.mppi import MPPI

from env.panda_pushing_env import TARGET_POSE_FREE_BOX, TARGET_POSE_OBSTACLES_BOX, OBSTACLE_CENTRE_BOX, OBSTACLE_HALFDIMS, BOX_SIZE

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE_BOX, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES_BOX, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE_BOX, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]

class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.target_state = None
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = torch.eye(env.action_space.shape[0])
        lambda_value = 1.0

        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)


    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        next_state = self.model(state, action)
        # print(next_state.shape,'next_state')
        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        state_tensor = torch.from_numpy(state)
        action_tensor = self.mppi.command(state_tensor)
        action = action_tensor.detach().numpy()
        # ---
        return action

    def set_hyper(self, hyperparameters):
        # ---
        self.mppi.set_hyper(hyperparameters)

    def get_cost(self):
        return self.mppi.get_cost()




def free_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    x = state
    Q = torch.diag(torch.tensor([1, 1, 0.1], dtype=state.dtype, device=state.device))
    cost = (x - target_pose) @ Q @ (x - target_pose).T #100x100
    cost = cost.diagonal()  
    # print(cost.shape)
    # ---
    return cost


def collision_detection(state):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    box_size = BOX_SIZE  # scalar for parameter w
    in_collision = None
    # --- Your code here
    # tilde 1 for box, tilde 2 for obstacle
    in_collision = [] 
    x2 = obstacle_centre[0]
    y2 = obstacle_centre[1]
    r_x1 = r_y1 = 0.5 * box_size
    r_x2 = 0.5 * obstacle_dims[0]
    r_y2 = 0.5 * obstacle_dims[1]
    x1 = state[:, 0]
    y1 = state[:, 1]
    theta = state[:, 2]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    delta_axis_x1 = torch.abs((x2 - x1) * cos_theta - (y1 - y2) * sin_theta)
    sum_r_axis_x1 = r_x1 + r_x2 * cos_theta
    delta_axis_x2 = torch.abs(x1 - x2)
    sum_r_axis_x2 = r_x2 + r_x1 * cos_theta
    delta_axis_y1 = torch.abs((x2 - x1) * sin_theta + (y1 - y2) * cos_theta)
    sum_r_axis_y1 = r_y1 + r_y2 * cos_theta
    delta_axis_y2 = torch.abs(y1 - y2)
    sum_r_axis_y2 = r_y2 + r_y1 * cos_theta
    in_collision_1 = (delta_axis_x1 < sum_r_axis_x1).float()
    in_collision_2 = (delta_axis_y1 < sum_r_axis_y1).float()
    in_collision_3 = (delta_axis_x2 < sum_r_axis_x2).float()
    in_collision_4 = (delta_axis_y2 < sum_r_axis_y2).float()
    in_collision = in_collision_1 * in_collision_2 * in_collision_3 * in_collision_4
    # ---
    return in_collision


def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    x = state
    Q = torch.tensor([[1,0,0],[0,1,0],[0,0,0.1]], dtype=state.dtype, device=state.device)
    cost = (x - target_pose) @ Q @ (x - target_pose).T #100x100
    cost = cost.diagonal()
    cost = cost + 100 * collision_detection(x)
    # ---
    return cost