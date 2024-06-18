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
        log_points = np.linspace(0, num_total_batches, self.num_images + 1, endpoint=False).astype(int)[1:] if self.num_images > 1 else [num_total_batches - 2]
        if batch_idx in log_points:
            if isinstance(trainer.logger, TensorBoardLogger):
                self.log_images_tensorboard(
                    logger=trainer.logger, 
                    inputs=outputs['input'], 
                    labels=batch['target'], 
                    preds_rgb=outputs['rgb_output'], 
                    preds_rgb2=outputs["rgb_outputs2"] if "rgb_outputs2" in outputs else outputs['rgb_output'],
                    preds_alpha=outputs['alpha_output'],
                    preds_reconstructed=outputs['reconstructed'],
                    stage='val', 
                    global_step=trainer.global_step
                )
            elif isinstance(trainer.logger, WandbLogger):
                self.log_images_wandb(
                    logger=trainer.logger, 
                    inputs=outputs['input'], 
                    labels=batch['target'] if "target" in batch else None, 
                    preds_rgb=outputs['rgb_output'],
                    preds_rgb2=outputs["rgb_outputs2"] if "rgb_outputs2" in outputs else outputs['rgb_output'],
                    preds_alpha=outputs['alpha_output'],
                    preds_reconstructed=outputs['reconstructed'],
                    segmentation=outputs["segmentation"] if "segmentation" in outputs else None,
                    segmentation_gt = outputs["segmentation_gt"] if "segmentation_gt" in outputs else None,
                    stage='val', 
                    global_step=trainer.global_step
                )

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

    def log_images_wandb(self, logger, inputs, labels, preds_rgb, preds_rgb2, preds_alpha, preds_reconstructed, stage, global_step, segmentation = None, segmentation_gt = None):
        import wandb
        import torchvision.utils as vutils

        
        # Create a grid of input images
        grid_inputs = vutils.make_grid(inputs)
        grid_inputs = grid_inputs.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
        wandb_inputs = wandb.Image(grid_inputs, caption=f'{stage}/inputs')


        
        grid_preds = vutils.make_grid(preds_rgb2)
        grid_preds = grid_preds.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
        wandb_preds2 = wandb.Image(grid_preds, caption=f'{stage}/preds_rgb2')

        # Create a grid of label images (assuming labels are single-channel)
        if labels is not None:
            grid_labels = vutils.make_grid(labels)
            grid_labels = grid_labels.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
            wandb_labels = wandb.Image(grid_labels, caption=f'{stage}/labels')
        else:
            wandb_labels = None

        if segmentation is not None:
            # Create a grid of prediction images (assuming preds are single-channel)
            grid_preds = vutils.make_grid(segmentation)
            grid_preds = grid_preds.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
            wandb_segmentation = wandb.Image(grid_preds, caption=f'{stage}/segmentation')

        if segmentation_gt is not None:
            # Create a grid of prediction images (assuming preds are single-channel)
            grid_preds = vutils.make_grid(segmentation_gt)
            grid_preds = grid_preds.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
            wandb_segmentation_gt = wandb.Image(grid_preds, caption=f'{stage}/segmentation_gt')
        else:
            wandb_segmentation_gt = None
            

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
            f'{stage}/predictions_rgb2': wandb_preds2,
            f'{stage}/predictions_alpha': wandb_alpha,
            f'{stage}/predictions_segmentation': wandb_segmentation,
            f'{stage}/wandb_segmentation_gt': wandb_segmentation_gt,
            f'{stage}/reconstructed': wandb_reconstructed,
            'global_step': global_step
        })