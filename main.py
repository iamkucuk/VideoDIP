import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from video_dip.callbacks.image_logger import ImageLogger
from video_dip.models.modules.relight import RelightVDPModule
from video_dip.data.datamodule import VideoDIPDataModule
from video_dip.models.optical_flow import RAFT, RAFTModelSize, Farneback
from pytorch_lightning.callbacks import LearningRateMonitor

# Initialize the model
model = RelightVDPModule(learning_rate=2e-3, loss_weights=[1, .02])

# Initialize the data module
data_module = VideoDIPDataModule(
    # input_path="outdoor_png/input/pair76", 
    # target_path="outdoor_png/GT/pair76",
    input_path='datasets/input/pair1',
    target_path='datasets/GT/pair1',
    flow_path="flow_outputs",
    batch_size=4, 
    num_workers=4
)

# data_module.dump_optical_flow(flow_model=RAFT(RAFTModelSize.LARGE))
# raise NotImplementedError

# Initialize the TensorBoard logger
tensorboard_logger = TensorBoardLogger("tb_logs", name="my_model")
wandb_logger = WandbLogger(project="video_dip_relighting")
wandb_logger.watch(model)

# Initialize the trainer with the logger
trainer = pl.Trainer(
    logger=wandb_logger, 
    devices=[1], 
    max_epochs=100, 
    callbacks=[
        ImageLogger(num_images=1),
        LearningRateMonitor(logging_interval='epoch')  # Log learning rate at every training step
    ],
    benchmark=True,
    num_sanity_val_steps=0,
    # sync_batchnorm=True
)

# Fit the model
trainer.fit(model, datamodule=data_module)
