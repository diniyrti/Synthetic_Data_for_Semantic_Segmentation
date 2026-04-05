import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

class SentinelUNet(pl.LightningModule):
    def __init__(self, in_channels=10, out_channels=16, learning_rate=1e-3): # Ubah out_channels sesuai kelasmu
        super().__init__()
        self.save_hyperparameters() 
        self.learning_rate = learning_rate
        
        #UNet Architecture
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        
        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Loss Function & Metrics
        self.loss_fn = smp.losses.DiceLoss(mode='multiclass')
        
        # Metrics for CSVLogger (IoU and Accuracy)
        self.iou_metric = MulticlassJaccardIndex(num_classes=out_channels)
        self.acc_metric = MulticlassAccuracy(num_classes=out_channels)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        
        # For multi-class, y must be of type long [Batch, H, W]
        y = y.long()
        
        # Predictions are taken from the highest channel index
        preds = torch.argmax(y_hat, dim=1)
            
        loss = self.loss_fn(y_hat, y)
        iou = self.iou_metric(preds, y)
        acc = self.acc_metric(preds, y)
        
        return loss, iou, acc

    def training_step(self, batch, batch_idx):
        loss, iou, acc = self._shared_step(batch)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mIoU", iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, iou, acc = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mIoU", iou, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, iou, acc = self._shared_step(batch)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_mIoU", iou, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)