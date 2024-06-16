import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from video_dip.models.modules.relight import RelightVDPModule
from video_dip.data.datamodule import VideoDIPDataModule
from video_dip.models.optical_flow.raft import RAFT, RAFTModelSize

# Initialize the model
model = RelightVDPModule()

# Initialize the data module
data_module = VideoDIPDataModule(
    input_path="datasets/input/pair1", 
    target_path="datasets/GT/pair1",
    flow_model=None, #RAFT(RAFTModelSize.LARGE),
    batch_size=4, 
    num_workers=4
)

# Initialize the TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="relight_vdp")

# Initialize the trainer with the logger
trainer = pl.Trainer(logger=logger, devices=1, max_epochs=10, precision='bf16')

# Fit the model
trainer.fit(model, datamodule=data_module)
