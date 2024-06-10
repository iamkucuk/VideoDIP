import torch
from torch import nn
from torchvision.models import vgg16

class PerceptualLoss(nn.Module):
    """
    Perceptual loss module using a pretrained VGG16 network.
    
    Attributes:
        vgg (nn.Module): Pretrained VGG16 network truncated at the 16th layer.
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg16(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """
        Calculate the perceptual loss between two tensors.

        Args:
            x (torch.Tensor): Predicted tensor.
            y (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Perceptual loss.
        """
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return torch.mean((x_features - y_features) ** 2)
