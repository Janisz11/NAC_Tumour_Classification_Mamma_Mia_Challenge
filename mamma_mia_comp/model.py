# model.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

# -------------------------------------------------
# 1. 3-D Spatial-Attention (CBAM-style)
# -------------------------------------------------
class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        att = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att                               # [B,C,D,H,W]


# -------------------------------------------------
# 2. 3-D ResNet-18 + Spatial Attention
# -------------------------------------------------
class ResNet3D_CBAM(nn.Module):
    def __init__(self,
                 in_channels: int = 6,
                 pretrained: bool = True,
                 freeze_until: str | None = "layer1"):
        super().__init__()

        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        net = r3d_18(weights=weights)

        # podmiana pierwszej konwolucji (3 → 6 kanałów)
        old_conv = net.stem[0]
        net.stem[0] = nn.Conv3d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                net.stem[0].weight[:, :3] = old_conv.weight
                if in_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    net.stem[0].weight[:, 3:] = mean_w.repeat(1, in_channels - 3, 1, 1, 1)

        # opcjonalne zamrożenie wczesnych warstw
        if freeze_until is not None:
            for name, p in net.named_parameters():
                if not name.startswith(freeze_until):
                    p.requires_grad = False

        # dopięcie SpatialAttention do głębszych bloków
        self.stem    = net.stem
        self.layer1  = net.layer1
        self.layer2  = nn.Sequential(net.layer2, SpatialAttention3D())
        self.layer3  = nn.Sequential(net.layer3, SpatialAttention3D())
        self.layer4  = nn.Sequential(net.layer4, SpatialAttention3D())
        self.avgpool = net.avgpool

        # projekcja 512 → 128
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)          # [B,512,1,1,1]
        return self.proj(x)          # [B,128]


# -------------------------------------------------
# 3. FusionPCRNet  (ResNet + TIC)
# -------------------------------------------------
class FusionPCRNet(nn.Module):
    def __init__(self, img_channels: int = 6, tic_dim: int = 4):
        super().__init__()

        self.cnn = ResNet3D_CBAM(img_channels)

        # TIC branch: 4 → 64  (LayerNorm ≠ zależny od size batch)
        self.fc_tic = nn.Sequential(
            nn.Linear(tic_dim, 32),
            nn.LayerNorm(32),        # ← BN → LN
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.LayerNorm(64),        # ← BN → LN
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Head: 128 + 64  → 128 → 2
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x_img: torch.Tensor, x_tic: torch.Tensor) -> torch.Tensor:
        img_vec = self.cnn(x_img)       # [B,128]
        tic_vec = self.fc_tic(x_tic)    # [B,64]
        x = torch.cat([img_vec, tic_vec], dim=1)
        return self.head(x)             # [B,2]
