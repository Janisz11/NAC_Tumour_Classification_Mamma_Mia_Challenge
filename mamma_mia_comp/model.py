import torch.nn as nn
import torch


class Simple3DCNN(nn.Module):
    """CNN zwracający wektor cech 128-D"""
    def __init__(self, in_channels=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward_features(self, x):
        return self.fc(self.conv(x))  # [B, 128]

    def forward(self, x):
        raise RuntimeError("Użyj FusionPCRNet do pełnej predykcji")


class FusionPCRNet(nn.Module):
    """Łączy cechy z CNN i cztery cechy TIC"""
    def __init__(self, img_channels=6, tic_dim=4):
        super().__init__()
        self.cnn = Simple3DCNN(img_channels)

        self.fc_tic = nn.Sequential(
            nn.Linear(tic_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x_img, x_tic):
        img_vec = self.cnn.forward_features(x_img)   # [B,128]
        tic_vec = self.fc_tic(x_tic)                 # [B,64]
        x = torch.cat([img_vec, tic_vec], dim=1)     # [B,192]
        return self.head(x)                          # [B,2]
