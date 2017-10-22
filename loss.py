from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F


def dice_coef(input, target, threshold=None):
    smooth = 1.0
    input_flatten = input.view(-1)
    if threshold is not None:
        input_flatten = (input_flatten > threshold).float()
    target_flatten = target.view(-1)
    intersection = (input_flatten * target_flatten).sum()
    return (
        (2. * intersection + smooth) /
        (input_flatten.sum() + target_flatten.sum() + smooth)
    )


class DiceLoss(nn.Module):
    def __init__(self, log=False):
        super().__init__()
        self.log = log

    def forward(self, input, target):
        dice_coef_value = dice_coef(F.sigmoid(input), target)
        if self.log:
            return -torch.log(dice_coef_value)
        else:
            return 1 - dice_coef_value


class BCEDiceLoss(nn.Module):
    def __init__(self, log_dice=False):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(log=log_dice)

    def forward(self, input, target):
        return self.bce_loss(input, target) + self.dice_loss(input, target)


losses = {
    'bce': nn.BCEWithLogitsLoss,
    'bce_dice': partial(BCEDiceLoss, log_dice=False),
    'bce_log_dice': partial(BCEDiceLoss, log_dice=True),
}
