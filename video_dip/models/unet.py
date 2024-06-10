import torch.nn as nn
from torch.nn.functional import interpolate

class UNet(nn.Module):
    """
    U-Net architecture used for both RGB-Net and α-Net.
    
    Attributes:
        encoder (nn.Sequential): Encoder part of the U-Net.
        decoder (nn.Sequential): Decoder part of the U-Net.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the U-Net with given input and output channels.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(UNet, self).__init__()
        
        # Encoder part of U-Net
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),  # Convolutional layer with 64 filters
            nn.BatchNorm2d(64),  # Batch normalization
            nn.LeakyReLU(0.2),  # Leaky ReLU activation
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # Max pooling layer to downsample
        )
        
        # Decoder part of U-Net
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Sigmoid activation for the final layer
        )

    def forward(self, x):
        """
        Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.encoder(x)
        x = interpolate(x, scale_factor=2, mode='bicubic')  # Upsampling using bicubic interpolation
        x = self.decoder(x)
        return x
