import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import numpy as np

class ImageLogger(pl.Callback):
    def __init__(self, num_images=4):
        super().__init__()
        self.num_images = num_images

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: torch.Tensor | torch.Dict[str, torch.Any] | None, batch: torch.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        num_total_batches = len(trainer.val_dataloaders)
        log_points = np.linspace(0, num_total_batches, self.num_images + 1, endpoint=False).astype(int)[1:]
        log_points = [num_total_batches - 2] if self.num_images == 1 else log_points
        if batch_idx in log_points:
            if isinstance(trainer.logger, TensorBoardLogger):
                self.log_images_tensorboard(
                    logger=trainer.logger, 
                    inputs=outputs['input'], 
                    labels=batch['target'], 
                    preds_rgb=outputs['rgb_output'], 
                    preds_alpha=outputs['alpha_output'],
                    preds_reconstructed=outputs['reconstructed'],
                    stage='val', 
                    global_step=trainer.global_step
                )
            elif isinstance(trainer.logger, WandbLogger):
                self.log_images_wandb(
                    logger=trainer.logger, 
                    inputs=outputs['input'], 
                    labels=batch['target'], 
                    preds_rgb=outputs['rgb_output'],
                    preds_alpha=outputs['alpha_output'],
                    preds_reconstructed=outputs['reconstructed'],
                    stage='val', 
                    global_step=trainer.global_step
                )
            

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     self.log_images(trainer, pl_module, 'val')

    # def on_test_epoch_end(self, trainer, pl_module):
    #     self.log_images(trainer, pl_module, 'test')

    # def log_images(self, trainer, pl_module, stage):
    #     # Get the last batch from the validation or test dataloader
    #     dataloader = trainer.val_dataloaders.dataset if stage == 'val' else trainer.test_dataloaders.dataset
    #     last_batch = next(iter(dataloader))
    #     inputs, labels = last_batch
    #     inputs = inputs[:self.num_images]
    #     labels = labels[:self.num_images]
        
    #     # Make sure to move inputs and labels to the same device as the model
    #     inputs = inputs.to(pl_module.device)
    #     labels = labels.to(pl_module.device)
        
    #     # Get the model predictions
    #     pl_module.eval()
    #     with torch.no_grad():
    #         preds = pl_module(inputs)
        
    #     # Log images
    #     if isinstance(trainer.logger, TensorBoardLogger):
    #         self.log_images_tensorboard(trainer.logger, inputs, labels, preds, stage, trainer.global_step)
    #     elif isinstance(trainer.logger, WandbLogger):
    #         self.log_images_wandb(trainer.logger, inputs, labels, preds, stage, trainer.global_step)

    def log_images_tensorboard(self, logger, inputs, labels, preds_rgb, preds_alpha, preds_reconstructed, stage, global_step):
        import torchvision.utils as vutils

        grid = vutils.make_grid(inputs)
        logger.experiment.add_image(f'{stage}/inputs', grid, global_step)

        grid = vutils.make_grid(labels.unsqueeze(1))  # Assuming labels are single-channel
        logger.experiment.add_image(f'{stage}/labels', grid, global_step)

        grid = vutils.make_grid(preds_rgb.unsqueeze(1))  # Assuming preds are single-channel
        logger.experiment.add_image(f'{stage}/predictions', grid, global_step)

        grid = vutils.make_grid(preds_alpha.unsqueeze(1))  # Assuming preds are single-channel
        logger.experiment.add_image(f'{stage}/alpha', grid, global_step)

        grid = vutils.make_grid(preds_reconstructed)
        logger.experiment.add_image(f'{stage}/reconstructed', grid, global_step)

    def log_images_wandb(self, logger, inputs, labels, preds_rgb, preds_alpha, preds_reconstructed, stage, global_step):
        import wandb
        import torchvision.utils as vutils

        # Create a grid of input images
        grid_inputs = vutils.make_grid(inputs)
        grid_inputs = grid_inputs.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
        wandb_inputs = wandb.Image(grid_inputs, caption=f'{stage}/inputs')

        # Create a grid of label images (assuming labels are single-channel)
        grid_labels = vutils.make_grid(labels)
        grid_labels = grid_labels.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
        wandb_labels = wandb.Image(grid_labels, caption=f'{stage}/labels')

        # Create a grid of prediction images (assuming preds are single-channel)
        grid_preds = vutils.make_grid(preds_rgb)
        grid_preds = grid_preds.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
        wandb_preds = wandb.Image(grid_preds, caption=f'{stage}/predictions')

        # Create a grid of alpha images (assuming preds are single-channel)
        grid_alpha = vutils.make_grid(preds_alpha)
        grid_alpha = grid_alpha.permute(1, 2, 0).cpu().float().numpy()
        wandb_alpha = wandb.Image(grid_alpha, caption=f'{stage}/alpha')

        grid_reconstructed = vutils.make_grid(preds_reconstructed)
        grid_reconstructed = grid_reconstructed.permute(1, 2, 0).cpu().float().numpy()
        wandb_reconstructed = wandb.Image(grid_reconstructed, caption=f'{stage}/reconstructed')

        # Log the images to wandb
        logger.experiment.log({
            f'{stage}/inputs': wandb_inputs,
            f'{stage}/labels': wandb_labels,
            f'{stage}/predictions_rgb': wandb_preds,
            f'{stage}/predictions_alpha': wandb_alpha,
            f'{stage}/reconstructed': wandb_reconstructed,
            'global_step': global_step
        })