import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

def collect_data_random(env, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = None
    # --- Your code here
    collected_data = []
    for i in range(num_trajectories):
        states = np.zeros((trajectory_length + 1, env.observation_space.shape[0]), dtype=np.float32)
        actions = np.zeros((trajectory_length, env.action_space.shape[0]), dtype=np.float32)
        state = env.reset()
        states[0] = state.astype(np.float32)
        for t in range(trajectory_length):
            action = env.action_space.sample()
            actions[t] = action.astype(np.float32)
            next_state, _, done, _ = env.step(action)
            states[t+1] = next_state.astype(np.float32)
            # if done:
            #     break
        trajectory = {'states': states, 'actions': actions}
        collected_data.append(trajectory)


    # ---
    return collected_data


def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here
    single_step_dataset = SingleStepDynamicsDataset(collected_data)
    train_set, val_set = random_split(single_step_dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # ---


    # ---
    return train_loader, val_loader


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'state': x_t,
     'action': u_t, ..., u_{t+num_steps-1},
     'next_state': x_{t+1}, ... , x_{t+num_steps}
    }
    where:
     state: torch.float32 tensor of shape (batch_size, state_size)
     next_state: torch.float32 tensor of shape (batch_size, num_steps, action_size)
     action: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # # --- Your code here
    multi_step_dataset = MultiStepDynamicsDataset(collected_data)
    train_set, val_set = random_split(multi_step_dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # # ---
    return train_loader, val_loader


class SingleStepDynamicsDataset(Dataset):
    """
    Each data sample is a dictionary containing (x_t, u_t, x_{t+1}) in the form:
    {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (state_size,)
     u_t: torch.float32 tensor of shape (action_size,)
     x_{t+1}: torch.float32 tensor of shape (state_size,)
    """

    def __init__(self, collected_data):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0]

    def __len__(self):
        return len(self.data) * self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None,
        }
        # --- Your code here
        idx_data = item // self.trajectory_length
        idx_traj = item % self.trajectory_length
        sample["state"] = torch.from_numpy(self.data[idx_data]["states"][idx_traj])
        sample["action"] = torch.from_numpy(self.data[idx_data]["actions"][idx_traj])
        sample["next_state"] = torch.from_numpy(self.data[idx_data]["states"][idx_traj+1])
        # assert sample["state"].dtype == torch.float32

        # ---
        return sample


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, collected_data, num_steps=4):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None
        }
        # --- Your code here
        idx_data = item // self.trajectory_length
        idx_traj = item % self.trajectory_length
        sample["state"] = torch.from_numpy(self.data[idx_data]["states"][idx_traj])
        sample["action"] = torch.from_numpy(self.data[idx_data]["actions"][idx_traj:idx_traj+self.num_steps])
        sample["next_state"] = torch.from_numpy(self.data[idx_data]["states"][idx_traj+1:idx_traj+1+self.num_steps])
        # assert sample["state"].dtype == torch.float32

        # ---
        return sample


class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}

    """

    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = block_width
        self.l = block_length

    def forward(self, pose_pred, pose_target):
        se2_pose_loss = None
        # --- Your code here
        rg = np.sqrt((self.w**2 + self.l**2)/12)
        se2_pose_loss = F.mse_loss(pose_pred[:, 0], pose_target[:, 0]) \
                        + F.mse_loss(pose_pred[:, 1], pose_target[:, 1]) \
                        + rg * F.mse_loss(pose_pred[:, 2], pose_target[:, 2])

        # ---
        return se2_pose_loss.sum()


class SingleStepLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        This method will perform a prediction with the model and computes the loss using the above loss function.
        """
        single_step_loss = None
        # --- Your code here
        pred_state = model(state, action)
        single_step_loss = self.loss(pred_state, target_state)

        # ---
        return single_step_loss


class MultiStepLoss(nn.Module):

    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        train a residual dynamics model using a recursive multistep loss
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = None
        # --- Your code here
        num_steps = actions.shape[1]
        multi_step_loss = 0.
        state_in = state
        # print(state.shape, actions.shape, target_states.shape)
        for step in range(num_steps):
            pred_state = model(state_in, actions[:, step])
            multi_step_loss += self.loss(pred_state, target_states[:, step]) * self.discount ** step
            state_in = pred_state

        # ---
        return multi_step_loss


class AbsoluteDynamicsModel(nn.Module):
    """
    Model the absolute dynamics x_{t+1} = f(x_{t},a_{t})
    The model architecture will be a 3 linear layer NN with hidden sizes of 100 and ReLU activations.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here
        self.hidden_dim = 100

        self.layer1 = nn.Linear(self.state_dim + self.action_dim, self.hidden_dim)

        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.layer3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.layer4 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.layer3 = nn.Linear(self.hidden_dim, state_dim)


        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here
        next_state = F.relu(self.layer1(torch.cat((state, action), dim=-1)))
        next_state = F.relu(self.layer2(next_state))
        # next_state = F.relu(self.layer3(next_state))
        # next_state = F.relu(self.layer4(next_state))

        next_state = self.layer3(next_state)


        # ---
        return next_state


class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here
        self.residual_dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, state_dim),
        )
        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here

        state_action = torch.cat([state, action], dim=-1)
        residual = self.residual_dynamics(state_action)
        next_state = state + residual

        # ---
        return next_state


# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
# --- Your code here
def train_dynamics_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs):
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for _ in pbar:
        train_loss = 0.
        model.train()
        for item in train_loader:
            optimizer.zero_grad()
            state = item["state"]
            action = item["action"]
            gt_next_state = item["next_state"]
            loss = loss_fn(model, state, action, gt_next_state)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss = 0.
        model.eval()
        for item in val_loader:
            state = item["state"]
            action = item["action"]
            gt_next_state = item["next_state"]
            with torch.no_grad():
                loss = loss_fn(model, state, action, gt_next_state)
            val_loss += loss.item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        pbar.set_description(f"Train Loss: {train_loss:.5f} | Validation Loss: {val_loss:.5f}")

    return train_losses, val_losses


# Backup for dynamics model training
# Train the dynamics model
# Load the collected data: 
model_path = 'assets/pretrained_models/'
collected_data = np.load(os.path.join(model_path, 'box_pushing_collected_data.npy'), allow_pickle=True)

pushing_multistep_residual_dynamics_model = ResidualDynamicsModel(3, 3)
train_loader, val_loader = process_data_multiple_step(collected_data, batch_size=500)

pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
pose_loss = MultiStepLoss(pose_loss, discount=0.9)

LR = 2e-4
NUM_EPOCHS = 1500
model = pushing_multistep_residual_dynamics_model
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
train_losses, val_losses = train_dynamics_model(model, train_loader, val_loader, optimizer, pose_loss, NUM_EPOCHS)

loss_target = 5e-4
print(f"Training loss less than {loss_target}? {train_losses[-1]<loss_target}")
print(f"Validation loss less than {loss_target}? {val_losses[-1]<loss_target}")

# Plot train loss and test loss:
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
axes[0].plot(train_losses)
axes[0].grid()
axes[0].set_title('Train Loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Train Loss')
axes[0].set_yscale('log')
axes[1].plot(val_losses)
axes[1].grid()
axes[1].set_title('Validation Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Validation Loss')
axes[1].set_yscale('log')
plt.show()

# save model:
save_path = os.path.join(model_path, 'pushing_multi_step_residual_dynamics_model.pt')
torch.save(pushing_multistep_residual_dynamics_model.state_dict(), save_path)

# ---
# ============================================================
