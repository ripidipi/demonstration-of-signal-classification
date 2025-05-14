import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size,
                                   padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))


class ResSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, reduction=16):
        super().__init__()
        self.conv1 = SeparableConv1d(in_ch, out_ch, kernel_size, padding)
        self.conv2 = SeparableConv1d(out_ch, out_ch, kernel_size, padding)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # → (B, C, 1)
            nn.Flatten(),              # → (B, C)
            nn.Linear(out_ch, out_ch//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch//reduction, out_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        w = self.se(out).unsqueeze(-1)  # → (B, C, 1)
        out = out * w + identity
        return F.relu(out, inplace=True)


class TimeTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, dropout=0.4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads,
            dim_feedforward=dim*2, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        # x: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)           # → (B, T, C)
        return x.permute(0, 2, 1)         # → (B, C, T)


class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            ResSEBlock(2,   64, kernel_size=3, padding=1),
            nn.MaxPool1d(2)  # → (B, 64, L/2)
        )
        self.layer2 = nn.Sequential(
            ResSEBlock(64, 128, kernel_size=3, padding=1),
            nn.MaxPool1d(2)  # → (B,128, L/4)
        )
        self.layer3 = nn.Sequential(
            ResSEBlock(128,256, kernel_size=3, padding=1),
            nn.MaxPool1d(2)  # → (B,256, L/8)
        )
        self.layer4 = nn.Sequential(
            ResSEBlock(256, 512, kernel_size=3, padding=1),
            nn.MaxPool1d(2)  # → (B,512, L/16)
        )

        self.transformer = TimeTransformer(dim=512, num_heads=8, num_layers=3, dropout=0.5)

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # → (B,256,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),         # → (B,256)
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.transformer(x)
        x = self.global_pool(x)
        return self.classifier(x)  # → (B,num_classes)
