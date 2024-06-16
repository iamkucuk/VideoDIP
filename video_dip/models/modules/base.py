import pytorch_lightning as pl
import torch
import torchvision

from video_dip.models.unet import UNet
from video_dip.losses import ReconstructionLoss, OpticalFlowWarpLoss

class VDPModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, loss_weights=[1, .02]):
        super().__init__()
        self.rgb_net = UNet()  # RGB-Net with 3 input and 3 output channels
        self.alpha_net = UNet()
        self.learning_rate = learning_rate

        self.reconstruction_loss = ReconstructionLoss()
        self.warp_loss = OpticalFlowWarpLoss()

        self.loss_weights = loss_weights

        # self.save_hyperparameters()

    def forward(self, img=None, flow=None):
        ret = {}
        if img is not None:
            ret['rgb'] = self.rgb_net(img)
        if flow is not None:
            ret['alpha'] = self.alpha_net(flow)
        return ret

    def reconstruction_fn(self, rgb_output, alpha_output):
        raise NotImplementedError("The reconstruction function is not implemented.")

    def inference(self, batch, batch_idx):
        input_frames = batch['input']
        flows = batch['flow']

        flow_frames = torchvision.utils.flow_to_image(flows) / 255.0

        output = self(img=input_frames, flow=flow_frames)
        rgb_output = output['rgb']
        alpha_output = output['alpha']

        reconstructed_frame = self.reconstruction_fn(rgb_output, alpha_output)

        return {
            "input": input_frames,
            "flow": flows,
            "reconstructed": reconstructed_frame,
            "rgb_output": rgb_output,
            "alpha_output": alpha_output
        }

    def training_step(self, batch, batch_idx):
        outputs = self.inference(batch, batch_idx)
        prev_alpha_output = self(flow=torchvision.utils.flow_to_image(batch['prev_flow']) / 255.0)['alpha'].detach()

        rec_loss = self.reconstruction_loss(outputs['input'], outputs['reconstructed'])
        warp_loss = self.warp_loss(outputs['flow'], prev_alpha_output, outputs['alpha_output'])

        loss = self.loss_weights[0] * rec_loss + self.loss_weights[1] * warp_loss

        self.log("train_loss", loss)
        self.log("rec_loss", rec_loss)
        self.log("warp_loss", warp_loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.inference(batch, batch_idx)

        # Add metrics here
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
