import cv2
import numpy as np
import torch
from torch import nn

class OpticalFlowLucasKanade(nn.Module):
    """
    Optical flow estimation using Lucas-Kanade method from OpenCV.
    """
    def __init__(self):
        super(OpticalFlowLucasKanade, self).__init__()

    def _farneback_one_image(self, image1, image2):
        """
        Estimate the optical flow between two images.

        Args:
            image1 (np.ndarray): First image (grayscale).
            image2 (np.ndarray): Second image (grayscale).

        Returns:
            np.ndarray: Estimated optical flow.
        """
        if (len(image1.shape) == 3) and (image1.shape[2] == 3):
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute magnite and angle of 2D vector
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv_mask = np.zeros_like(image1)
        hsv_mask[..., 1] = 255

        # Set image hue value according to the angle of optical flow
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        # Set value as per the normalized magnitude of optical flow
        hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to rgb
        rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
        # Convert flow to tensor
        flow_rgb = torch.tensor(rgb_representation).permute(2, 0, 1).unsqueeze(0).float().cuda()
        return flow_rgb
    
    def forward(self, image1, image2):
        """
        Estimate the optical flow between two images.

        Args:
            image1 (torch.Tensor): First image (grayscale).
            image2 (torch.Tensor): Second image (grayscale).

        Returns:
            torch.Tensor: Estimated optical flow.
        """
        image1 = image1.squeeze().cpu().numpy()
        image2 = image2.squeeze().cpu().numpy()

        # Convert images to uint8
        image1 = np.uint8(image1 * 255)
        image2 = np.uint8(image2 * 255)

        # Convert images to grayscale if they are RGB
        if len(image1.shape) == 4:
            flows = []
            for i in range(len(image1)):
                flows.append(self._farneback_one_image(image1[i], image2[i]))
            return torch.cat(flows)
        else:
            return self._farneback_one_image(image1, image2)

        
