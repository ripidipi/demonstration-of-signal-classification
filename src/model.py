import torch, torch.nn as nn, torch.nn.functional as F

class SeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, p):
        super().__init__()
        self.depth = nn.Conv1d(in_ch, in_ch, k, padding=p, groups=in_ch, bias=False)
        self.point = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm1d(out_ch)
    def forward(self,x): return F.relu(self.bn(self.point(self.depth(x))))

class ResSEBlock(nn.Module):
    def __init__(self,in_ch,out_ch,k,p,reduction=8):
        super().__init__()
        self.conv1 = SeparableConv1d(in_ch,out_ch,k,p)
        self.conv2 = SeparableConv1d(out_ch,out_ch,k,p)
        self.skip  = nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
        self.se    = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_ch, out_ch//reduction),
            nn.ReLU(),
            nn.Linear(out_ch//reduction, out_ch),
            nn.Sigmoid()
        )
    def forward(self,x):
        s = self.skip(x)
        o = self.conv2(self.conv1(x))
        w = self.se(o).unsqueeze(-1)
        return F.relu(o*w + s)

class TimeTransformer(nn.Module):
    def __init__(self,dim,heads=8,layers=3):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*2,
            dropout=0.1, activation='gelu', batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
    def forward(self,x):
        t = x.permute(0,2,1)
        t = self.enc(t)
        return t.permute(0,2,1)

class CNNClassifier(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            ResSEBlock(2,64,3,1),
            ResSEBlock(64,128,3,1),
            nn.MaxPool1d(2),
            ResSEBlock(128,256,3,1),
            nn.MaxPool1d(2),
            ResSEBlock(256,512,3,1)
        )
        self.trans = TimeTransformer(dim=512)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )
    def forward(self,x):
        x = self.stem(x)
        x = self.trans(x)
        x = self.pool(x)
        return self.fc(x)
