import torch
from . import VDPModule

class RelightVDPModule(VDPModule):
    """
    A module for relighting in VideoDIP.

    Args:
        learning_rate (float): The learning rate for optimization (default: 1e-3).
        loss_weights (list): The weights for different losses (default: [1, .02]).
    """

    def __init__(self, learning_rate=1e-3, loss_weights=[1, .02]):
        super().__init__(learning_rate, loss_weights)

        # Randomly initialize a parameter named gamma
        self.register_buffer("gamma_inv", torch.tensor(1.0))

        self.save_hyperparameters()

    def reconstruction_fn(self, rgb_output, alpha_output):
        """
        Reconstructs the output by performing element-wise multiplication of RGB layers with alpha layers.

        Args:
            rgb_output (torch.Tensor): The RGB output tensor.
            alpha_output (torch.Tensor): The alpha output tensor.

        Returns:
            torch.Tensor: The reconstructed output tensor.
        """
        return alpha_output * rgb_output ** self.gamma_inv
    
