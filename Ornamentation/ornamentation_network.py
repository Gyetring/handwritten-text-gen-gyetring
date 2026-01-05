import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Basic Blocks
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 由于 64×64 是 2 的整数次幂，理论上不需要 crop
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# =========================
# U-Net
# =========================
class OrnamentationUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.down4 = Down(512, 1024)
        self.up0   = Up(1024, 512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)

        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)      # 64×64
        x2 = self.down1(x1)   # 32×32
        x3 = self.down2(x2)   # 16×16
        x4 = self.down3(x3)   # 8×8

        x5 = self.down4(x4)
        x = self.up0(x5, x4)

        # x = self.up1(x4, x3)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return torch.sigmoid(self.outc(x))


# =========================
# Loss Function
# =========================
class OrnamentationLoss(nn.Module):
    """
    默认使用 L1 loss（更适合笔画宽度回归）
    """
    def __init__(self, lambda_l1=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1

    def forward(self, pred, target):
        return self.lambda_l1 * self.l1(pred, target)


# =========================
# Simple Test
# =========================
if __name__ == "__main__":
    model = OrnamentationUNet()
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    print("Output shape:", y.shape)
