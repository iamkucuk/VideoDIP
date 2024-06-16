import pytorch_lightning as pl
import torch

from video_dip.models.vdp import VDPModel
from video_dip.losses import ReconstructionLoss, OpticalFlowWarpLoss

class VDPModule(pl.LightningModule):

    def __init__(self, reconstruction_fn, learning_rate=1e-3, loss_weights=[1, .02]):
        super().__init__()

        self.model = VDPModel()
        self.learning_rate = learning_rate

        self.criterion_rec = ReconstructionLoss()
        self.criterion_warp = OpticalFlowWarpLoss()

        self.loss_weights = loss_weights

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_frames = batch['input']
        try:
            target_frames = batch['target']
        except KeyError:
            raise KeyError("The target frames are not provided.")
        flow_frames = batch['flow']

        output = self(input_frames)
        rgb_output = output['rgb']
        alpha_output = output['alpha']

        rec_loss = self.criterion_rec(x_hat=rgb_output, x=target_frames)
        warp_loss = self.criterion_warp(x_hat=alpha_output, x=target_frames, flow=flow_frames)

        loss = rec_loss + warp_loss

        self.log('train_loss', loss)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
