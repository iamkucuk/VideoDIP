import torch.nn as nn
from torch.nn.functional import interpolate

class UNet(nn.Module):
    def __init__(self, in_channels=3, channels=[64, 64, 96, 128, 128, 128, 128, 96]):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            self._conv(in_channels, channels[0]),
            self._conv(channels[0], channels[1]),
            nn.MaxPool2d(2),
            self._conv(channels[1], channels[2]),
            self._conv(channels[2], channels[3]),
            nn.MaxPool2d(2),
            self._conv(channels[3], channels[4]),
            self._conv(channels[4], channels[5]),
            self._conv(channels[5], channels[6]),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[6], channels[7], 4, 1, 0)
        )
        self.decoder = nn.Sequential(
            self._upconv(channels[7], channels[6], 4, 1, 0),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            self._conv(channels[6], channels[5]),
            self._conv(channels[5], channels[4]),
            self._conv(channels[4], channels[3]),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            self._conv(channels[3], channels[2]),
            self._conv(channels[2], channels[1]),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            self._conv(channels[1], channels[0]),
            nn.ConvTranspose2d(channels[0], in_channels, 3, 1, 1)
        )

    def _conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _upconv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        encoder_outputs = []
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)
        
        for idx, layer in enumerate(self.decoder):
            x = layer(x + encoder_outputs[-(idx + 1)])
        
        return x