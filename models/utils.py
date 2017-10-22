from functools import partial

import torch
import torch.nn as nn


kaiming_normal = partial(torch.nn.init.kaiming_normal, mode='fan_out')


weight_init_functions = {
    nn.Linear: kaiming_normal,
    nn.Conv2d: kaiming_normal,
    nn.BatchNorm2d: lambda w: w.data.fill_(1),
}


def weight_init(net, weight_init_functions=weight_init_functions):
    for m in net.modules():
        weight_init_fn = weight_init_functions.get(type(m))
        if weight_init_fn:
            weight_init_fn(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
