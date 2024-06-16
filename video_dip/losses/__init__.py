# import torch
from .reconstruction_loss import ReconstructionLoss
from .perceptual_loss import PerceptualLoss
from .optical_flow_warp_loss import OpticalFlowWarpLoss
from .flow_similarity_loss import FlowSimilarityLoss

# fismloss = FlowSimilarityLoss()
# i = 2
# b = 64
# # b*i --> 128
# masks = torch.randn(64,i,64,64)
# imgs = torch.randn(b,3,64,64)
# fsimloss_value = fismloss(masks, imgs)
# print(fsimloss_value.shape)