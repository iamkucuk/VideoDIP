import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torchvision

import numpy as np

class ImageLogger(pl.Callback):
    def __init__(self, num_images=4):
        super().__init__()
        self.num_images = num_images

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: torch.Tensor | torch.Dict[str, torch.Any] | None, batch: torch.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        num_total_batches = len(trainer.val_dataloaders)
        log_points = np.linspace(0, num_total_batches, self.num_images + 1, endpoint=False).astype(int)[1:] if self.num_images > 1 else [num_total_batches - 2]
        if batch_idx in log_points:
            flow_image = torchvision.utils.flow_to_image(batch['flow']) / 255.0
            if isinstance(trainer.logger, TensorBoardLogger):
                self.log_images_tensorboard(
                    logger=trainer.logger, 
                    inputs=outputs['input'], 
                    labels=batch['target'], 
                    preds_rgb=outputs['rgb_output'], 
                    preds_alpha=outputs['alpha_output'],
                    preds_reconstructed=outputs['reconstructed'],
                    flow=flow_image,
                    stage='val', 
                    global_step=trainer.global_step
                )
            elif isinstance(trainer.logger, WandbLogger):
                self.log_images_wandb(
                    logger=trainer.logger, 
                    inputs=outputs['input'], 
                    labels=batch['target'] if "target" in batch else None, 
                    preds_rgb=outputs['rgb_output'],
                    preds_alpha=outputs['alpha_output'],
                    preds_reconstructed=outputs['reconstructed'],
                    flow=flow_image,
                    stage='val', 
                    global_step=trainer.global_step
                )

    def log_images_tensorboard(self, logger, inputs, labels, preds_rgb, preds_alpha, preds_reconstructed, flow, stage, global_step):
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

        grid = vutils.make_grid(flow)
        logger.experiment.add_image(f'{stage}/flow', grid, global_step)

    def log_images_wandb(self, logger, inputs, labels, preds_rgb, preds_alpha, preds_reconstructed, flow, stage, global_step):
        import wandb
        import torchvision.utils as vutils

        # Create a grid of input images
        grid_inputs = vutils.make_grid(inputs)
        grid_inputs = grid_inputs.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
        wandb_inputs = wandb.Image(grid_inputs, caption=f'{stage}/inputs')

        # Create a grid of label images (assuming labels are single-channel)
        if labels:
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

        grid_flow = vutils.make_grid(flow)
        grid_flow = grid_flow.permute(1, 2, 0).cpu().float().numpy()
        wandb_flow = wandb.Image(grid_flow, caption=f'{stage}/flow')

        # Log the images to wandb
        logger.experiment.log({
            f'{stage}/inputs': wandb_inputs,
            f'{stage}/labels': wandb_labels if labels else None,
            f'{stage}/predictions_rgb': wandb_preds,
            f'{stage}/predictions_alpha': wandb_alpha,
            f'{stage}/reconstructed': wandb_reconstructed,
            f'{stage}/flow': wandb_flow,
            'global_step': global_step
        })