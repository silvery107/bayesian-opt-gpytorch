import torch
import gpytorch
import matplotlib.pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel


class RBF_GP(ExactGP):

    def __init__(self, train_x, train_y):
        likelihood = GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class Matern_GP(ExactGP):

    def __init__(self, train_x, train_y):
        likelihood = GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # print(f"mean_x: {mean_x.shape}, covar_x: {covar_x.shape}")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskGPModel(ExactGP):
    """
        Multi-task GP model for dynamics x_{t+1} = f(x_t, u_t)
        Each output dimension of x_{t+1} is represented with a seperate GP
    """

    def __init__(self, train_x, train_y, num_tasks):
        # We use a multitask Gaussian likelihood
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks,
                                                                    noise_constraint=gpytorch.constraints.GreaterThan(1e-9)
                                                                    )
        super().__init__(train_x, train_y, likelihood)
        input_shape = train_x.shape[1]
        batch_shape = train_y.shape[1]
        # print(f"input shape: {input_shape}, output/batch shape: {batch_shape}")
        self.mean_module = ResidualMean(torch.Size([batch_shape]))
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=input_shape,
                                                  batch_shape=torch.Size([batch_shape])),
                                        batch_shape=torch.Size([batch_shape]))
        # print(f"lengthscale shape: {self.covar_module.base_kernel.lengthscale.shape}")

    def forward(self, x):
        """

        Args:
            x: torch.tensor of shape (B, dx + du) concatenated state and action

        Returns: gpytorch.distributions.MultitaskMultivariateNormal - Gaussian prediction for next state

        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # print(mean_x.shape, covar_x.shape)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class ResidualMean(gpytorch.means.Mean):
    def __init__(self, batch_shape):
        super().__init__()
        self.batch_shape = batch_shape

    def forward(self, input):
        """
        Residual mean function
        Args:
            input: torch.tensor of shape (N, dx + du) containing state and control vectors

        Returns:
            mean: torch.tensor of shape (dx, N) containing state vectors

        """
        mean = input[:, :list(self.batch_shape)[0]].T
        return mean

def train_gp_hyperparams(model, likelihood, train_x, train_y, lr=0.1, iters=100):
    """
        Function which optimizes the GP Kernel & likelihood hyperparameters
    Args:
        model: gpytorch.model.ExactGP model
        likelihood: gpytorch likelihood
        train_x: (N, dx) torch.tensor of training inputs
        train_y: (N, dy) torch.tensor of training targets
        lr: Learning rate

    """

    training_iter = iters
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Find optimal model hyperparameters
    model.train() #Make sure to do this before training
    likelihood.train() #Make sure to do this before training

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        # print('Iter %2d/%d - Loss: %.3f   outputscale: %.3f   noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     # model.covar_module.base_kernel.lengthscale.item(),
        #     model.covar_module.outputscale.item(),
        #     model.likelihood.noise.item()
        # ))
        optimizer.step()


def plot_gp_predictions(model, likelihood, train_x, train_y, test_x, title):
    """
        Generates GP plots for GP defined by model & likelihood
    """
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    with torch.no_grad():
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        lower, upper = observed_pred.confidence_region()
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-10, 10])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_title(title)
