import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from video_dip.callbacks.image_logger import ImageLogger
from video_dip.models.modules.segmentation import SegmentationVDPModule
from video_dip.data.datamodule import VideoDIPDataModule
from video_dip.models.optical_flow.raft import RAFT, RAFTModelSize

# Initialize the model
model = SegmentationVDPModule(learning_rate=0.1)

# Initialize the data module
data_module = VideoDIPDataModule(
    input_path="datasets/input/bear", 
    target_path="datasets/GT/bear",
    flow_model=RAFT(RAFTModelSize.LARGE),
    flow_path="datasets/input/bear_flow",
    batch_size=2, 
    num_workers=8
)

# Initialize the TensorBoard logger
tensorboard_logger = TensorBoardLogger("tb_logs", name="my_model")
wandb_logger = WandbLogger(project="video_dip_segmentation")
wandb_logger.watch(model)

# Initialize the trainer with the logger
trainer = pl.Trainer(
    logger=wandb_logger, 
    devices=1, 
    max_epochs=100, 
    callbacks=[ImageLogger(num_images=1)]
    # sync_batchnorm=True
)

# Fit the model
trainer.fit(model, datamodule=data_module)
