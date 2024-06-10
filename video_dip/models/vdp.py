from torch import nn

from video_dip.models.unet import UNet

class VDPModel(nn.Module):
    """
    Video Decomposition Prior (VDP) model integrating RGB-Net and α-Net.
    
    Attributes:
        rgb_net (UNet): U-Net for predicting RGB layers.
        alpha_net (UNet): U-Net for predicting alpha (opacity) layers.
        learning_rate (float): Learning rate for the optimizer.
        vgg (nn.Module): Pretrained VGG16 model for perceptual loss calculation.
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
        rgb_layers = self.rgb_net(x)
        alpha_layers = self.alpha_net(x)
        return rgb_layers, alpha_layers