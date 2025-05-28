import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_att = self.fc(self.avg(x))
        max_att = self.fc(self.max(x))
        return x * (avg_att + max_att)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class EnhancedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expansion=4, drop_path=0.1):
        super().__init__()
        mid_ch = out_ch * expansion
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, mid_ch, 3, stride, padding=1, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.GELU(),
            ChannelAttention(mid_ch),
            nn.Conv1d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        self.att = ChannelAttention(out_ch)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        res = self.conv(x)
        sc = self.shortcut(x)
        out = res + sc
        return F.gelu(self.att(self.drop_path(out)))

class SOTAClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def make_backbone():
            return nn.Sequential(
                EnhancedResBlock(2, 64, stride=2, drop_path=0.05),
                EnhancedResBlock(64, 128, stride=2, drop_path=0.05),
                EnhancedResBlock(128, 256, stride=2, drop_path=0.1),
                EnhancedResBlock(256, 512, stride=2, drop_path=0.1),
                EnhancedResBlock(512, 512, stride=2, drop_path=0.1),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )

        self.backb1 = make_backbone()
        self.backb2 = make_backbone()

        self.const_fc = nn.Sequential(
            nn.Linear(64 * 64, 256), nn.GELU(), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.GELU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.GELU()
        )

        self.snr_fc = nn.Sequential(
            nn.Linear(1, 128), nn.GELU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.GELU()
        )

        fusion_dim = 512 + 512 + 64 + 64
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, iq1, iq2, const, snr):
        f1 = self.backb1(iq1)
        f2 = self.backb2(iq2)
        cfeat = self.const_fc(const.view(const.size(0), -1))
        sfeat = self.snr_fc(snr)
        x = torch.cat([f1, f2, cfeat, sfeat], dim=1)
        return self.classifier(x)