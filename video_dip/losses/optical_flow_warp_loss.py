import torch
from torch import nn

class OpticalFlowWarpLoss(nn.Module):
    """
    Optical flow warp loss module to ensure temporal coherence.
    """
    def __init__(self):
        super(OpticalFlowWarpLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, flow, x, x_hat):
        """
        Calculate the optical flow warp loss.

        Args:
            flow (torch.Tensor): Optical flow tensor.
            x (torch.Tensor): Ground truth tensor at time t-1.
            x_hat (torch.Tensor): Predicted tensor at time t.

        Returns:
            torch.Tensor: Optical flow warp loss.
        """
        warped_x = self.warp(x, flow)
        return self.l1_loss(warped_x, x_hat)
    
    @staticmethod
    def warp(x, flow):
        """
        Warp an image/tensor (x) according to the given flow.

        Args:
            x (torch.Tensor): Image tensor.
            flow (torch.Tensor): Optical flow tensor.

        Returns:
            torch.Tensor: Warped image tensor.
        """
        B, C, H, W = x.size()
        grid = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid = torch.stack(grid, dim=-1).float().cuda()
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        grid = grid + flow.permute(0, 2, 3, 1)
        grid[..., 0] = 2.0 * grid[..., 0] / (H - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (W - 1) - 1.0
        grid = grid.permute(0, 2, 3, 1)
        return nn.functional.grid_sample(x, grid)
