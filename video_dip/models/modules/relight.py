import torch
from . import VDPModule

class RelightVDPModule(VDPModule):
    def __init__(self, learning_rate=1e-3, loss_weights=[1, .02]):
        super().__init__(learning_rate, loss_weights)

        # Randomly initialize a parameter named gamma
        self.register_buffer("gamma_inv", torch.tensor(0.5))

        self.save_hyperparameters()

    def reconstruction_fn(self, rgb_output, alpha_output):
        # Element-wise multiplication of RGB layers with alpha layers
        return rgb_output * alpha_output ** self.gamma_inv
    
