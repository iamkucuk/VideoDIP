import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from video_dip.callbacks.image_logger import ImageLogger
from video_dip.models.modules.segmentation import SegmentationVDPModule
from video_dip.data.datamodule import VideoDIPDataModule
from video_dip.models.optical_flow.raft import RAFT, RAFTModelSize
from video_dip.models.optical_flow.farneback import Farneback
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging

from torch.optim.lr_scheduler import LinearLR
# Initialize the model
model = SegmentationVDPModule(
    learning_rate=5e-4, 
    loss_weights=[.001, 1 , 1, .01, .01],
    multi_step_scheduling_kwargs={
        'milestones': [3, 5, 45, 75],
        'gamma': .5,
    },
    warmup=True
    )

# Initialize the data module
data_module = VideoDIPDataModule(
    input_path="datasets/input/blackswan", 
    target_path="datasets/GT/blackswan",
    flow_path="datasets/input/blackswan_flow",
    batch_size=2, 
    num_workers=8
)


data_module.dump_optical_flow(flow_model=RAFT(RAFTModelSize.LARGE))
# raise NotImplementedError

# Initialize the TensorBoard logger
tensorboard_logger = TensorBoardLogger("tb_logs", name="my_model")
wandb_logger = WandbLogger(project="video_dip_segmentation")
wandb_logger.watch(model)

# Initialize the trainer with the logger
trainer = pl.Trainer(
    logger=wandb_logger, 
    devices=1, 
    max_epochs=100, 
    callbacks=[ImageLogger(num_images=1),
               LearningRateMonitor(logging_interval='epoch')  ],# Log learning rate at every training step
    benchmark=True,
    num_sanity_val_steps=0
    # sync_batchnorm=True
)

# Fit the model
trainer.fit(model, datamodule=data_module)
