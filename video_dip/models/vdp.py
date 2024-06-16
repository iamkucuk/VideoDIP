from torch import nn

from video_dip.models.unet import UNet

class VDPModel(nn.Module):
    """
    Video Decomposition Prior (VDP) model integrating RGB-Net and α-Net.
    
    Attributes:
        rgb_net (UNet): U-Net for predicting RGB layers.
        alpha_net (UNet): U-Net for predicting alpha (opacity) layers.
        learning_rate (float): Learning rate for the optimizer.
    """
    def __init__(self):
        """
        Initialize the VDP model with specified learning rate.

        Args:
            learning_rate (float): Learning rate for the optimizer. Default is 1e-3.
        """
        super(VDPModel, self).__init__()
        self.rgb_net = UNet(3, 3)  # RGB-Net with 3 input and 3 output channels
        self.alpha_net = UNet(3, 1)  # α-Net with 3 input and 1 output channel

    def forward(self, x):
        """
        Forward pass of the VDP model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: RGB layers and alpha layers.
        """
        rgb_output = self.rgb_net(x)
        alpha_output = self.alpha_net(x)
        return {'rgb': rgb_output, 'alpha': alpha_output}