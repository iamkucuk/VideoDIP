import pytorch_lightning as pl
import torch

from video_dip.models.vdp import VDPModel

class VDPModule(pl.LightningModule):

    def __init__(self, learning_rate=1e-3):
        super().__init__()

        self.model = VDPModel()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
