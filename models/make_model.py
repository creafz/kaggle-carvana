from functools import partial

from models.se_refinenet import SERefineNet
from models.utils import weight_init


MODELS = {
    'se_refinenet_1024': partial(SERefineNet, input_shape=(3, 1024),
                                 freeze_resnet=True, weight_init=weight_init),
    'se_refinenet_1024_all_layers_unfrozen': partial(SERefineNet,
                                                     input_shape=(3, 1024),
                                                     freeze_resnet=False),
}


def make_model(name, cuda=True):
    model = MODELS[name]()
    if cuda:
        model = model.cuda()
    return model
