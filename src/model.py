import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Basic Building Blocks
# -----------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B * N, C)
        x = self.net(x)
        return x.view(B, N, -1)


class RandLA_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp = MLPBlock(in_ch, out_ch)

    def forward(self, x):
        return self.mlp(x)


class KPConv_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp = MLPBlock(in_ch, out_ch)

    def forward(self, x):
        return self.mlp(x)


# -----------------------------
# Main Model
# -----------------------------
class RandLA_KPConv_UNet_Classifier(nn.Module):
    def __init__(
        self,
        in_channels=5,
        num_classes=5,
        enc1_out=64,
        enc2_out=128,
        bottleneck_ch=256,
        dropout=0.3,
        **kwargs
    ):
        super().__init__()

        # Encoder
        self.enc1 = RandLA_Block(in_channels, enc1_out)
        self.enc2 = KPConv_Block(enc1_out, enc2_out)

        # Bottleneck
        self.bottleneck = MLPBlock(enc2_out, bottleneck_ch)

        # Global pooling
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(bottleneck_ch, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, N, C]
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.bottleneck(x)

        # Global pooling
        x = x.permute(0, 2, 1)  # [B, C, N]
        x = self.pool(x).squeeze(-1)

        return self.fc(x)