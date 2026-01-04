import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_maps=64):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # Input: (N, 1, 64, 64)

            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (N, 64, 32, 32)

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (N, 128, 16, 16)

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (N, 256, 8, 8)

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (N, 512, 4, 4)

            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # -> (N, 1, 1, 1)
        )

    def forward(self, x):
        return self.net(x).view(-1)
