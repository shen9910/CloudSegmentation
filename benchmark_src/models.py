import utils
import pytorch_lightning as pl
import torch
import numpy as np
import random
import segmentation_models_pytorch as smp

class CloudModel(pl.LightningModule):
    def __init__(self,
                 dataloaders,
                 model,
                 optimizer,
                 scheduler,
                 config) -> None:
        super(CloudModel, self).__init__()
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.model = model
        self.include_steps = None
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    
    def forward(self, x):
        return self.model(x)
    
    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        self.val_length = len(self.dataloaders["val"])
        self.include_steps = random.sample(range(0, self.val_length), k=20)
        return self.dataloaders["val"]
    
    def training_step(self, batch, batch_idx):
        x, y = batch['chip'], batch['label'].long()
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['chip'], batch['label'].long()
        pred = self(x)
        pred = pred.sigmoid()
        pred = (pred > 0.5).float().squeeze()
        pred_np = pred.long().cpu().detach().numpy()
        y_np = y.long().cpu().detach().numpy()
        batch_iou = utils.jaccard_metric(pred_np, y_np)

        self.log(
            "iou_score", batch_iou, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return batch_iou
    
    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

