from torch import nn as nn
from torch.nn import functional as F


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block
    https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.channels = channels
        self.fc1 = nn.Linear(channels, channels // reduction,
                             bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels,
                             bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.avg_pool2d(x, kernel_size=x.size()[2:])
        out = self.fc1(out.view(out.size(0), -1))
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out.view(-1, self.channels, 1, 1)
