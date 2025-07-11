import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, down=True, act="relu", use_dropout=False
    ):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            (
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    4,
                    2,
                    1,
                    bias=False,
                    padding_mode="reflect",
                )
                if down
                else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(
            features, features * 2, down=True, act="leaky", use_dropout=False
        )
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )  # Fixed: Downsampling here

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                features * 8, features * 8, 4, 2, 1, padding_mode="reflect"
            ),  # Downsample here
            nn.ReLU(),
        )

        self.up1 = Block(
            features * 8, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(
            features * 2 * 2, features, down=False, act="relu", use_dropout=False
        )

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, in_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)  # Shape: (B, C, H/2, W/2)
        d2 = self.down1(d1)  # Shape: (B, C, H/4, W/4)
        d3 = self.down2(d2)  # Shape: (B, C, H/8, W/8)
        d4 = self.down3(d3)  # Shape: (B, C, H/16, W/16)
        d5 = self.down4(d4)  # Shape: (B, C, H/32, W/32)
        d6 = self.down5(d5)  # Shape: (B, C, H/64, W/64)
        d7 = self.down6(d6)  # Shape: (B, C, H/128, W/128)

        bottleneck = self.bottleneck(d7)  # Shape: (B, C, H/256, W/256)

        up1 = self.up1(bottleneck)  # Shape: (B, C, H/128, W/128)
        up2 = self.up2(torch.cat([up1, d7], dim=1))  # Shape: (B, C*2, H/128, W/128)
        up3 = self.up3(torch.cat([up2, d6], dim=1))  # Shape: (B, C*2, H/64, W/64)
        up4 = self.up4(torch.cat([up3, d5], dim=1))  # Shape: (B, C*2, H/32, W/32)
        up5 = self.up5(torch.cat([up4, d4], dim=1))  # Shape: (B, C*2, H/16, W/16)
        up6 = self.up6(torch.cat([up5, d3], dim=1))  # Shape: (B, C*2, H/8, W/8)
        up7 = self.up7(torch.cat([up6, d2], dim=1))  # Shape: (B, C*2, H/4, W/4)

        # Ensure up7 matches d1 before concatenation
        final_output = self.final_up(torch.cat([up7, d1], dim=1))  # Shape: (B, C, H, W)

        return final_output
