import torch
from torch import nn
from torchvision.models import vgg16

class FlowSimilarityLoss(nn.Module):
    """
    
    
    Attributes:
        
    """
    def __init__(self):
        super(FlowSimilarityLoss, self).__init__()
        self.vgg = vgg16(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, m, frgb):
        """
        Calculate the flow similarity loss.

        Args:
            M (torch.Tensor): b,i(ith layer),h,w Motion tensor of all video layers at time t(mask)
            F^RGB (torch.Tensor): b,c,h,w RGB image flow from t-1 to t

        Returns:
            torch.Tensor: Flow Similarity Loss
        """
        (b,i,h,w) = m.size()
        (b,c,h,w) = frgb.size()

        m = m.unsqueeze(2)
        frgb = frgb.unsqueeze(1)
        # their product will be b,i,c,h,w we need to get rid of the i in order to feed to the VGG
        # therefore put that dimension into the batch as well
        production = frgb * m
        neg_production = frgb * (1-m)

        product = self.vgg(production.view(b*i, c,h,w))
        neg_product = self.vgg(neg_production.view(b*i, c,h,w))
        lfsim = (product * neg_product) / (torch.norm(product) * torch.norm(neg_product))
        return lfsim
