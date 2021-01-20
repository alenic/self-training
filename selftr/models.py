import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as T


class CNNSuper(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4):
        super(CNNSuper, self).__init__()
        self.dropout_rate = dropout_rate
        self.block1 = self.block(3, 32)
        self.block2 = self.block(32, 64)
        self.fc = nn.Linear(64 * 7 * 7, 128)
        self.logits = nn.Linear(128, num_classes)
        self.init_weights()

    def block(self, in_channels, out_channels, groups=1):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        return block

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.logits.weight)

    def forward(self, x):
        x = self.block1(x)
        x = F.relu(x)
        x = self.block2(x)
        x = F.relu(x)
        x = x.flatten(1)
        x = self.fc(x)
        if self.train and self.dropout_rate > 0:
            x = F.dropout(x)
        x = self.logits(x)
        return x