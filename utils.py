from collections import defaultdict
import os
import random

import numpy as np
from pycrayon import CrayonClient
import torch

import config
from models.make_model import make_model


class MetricMonitor:
    def __init__(self, batch_size=None):
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {'sum': 0, 'count': 0, 'avg': 0})

    def update(self, metric_name, value, n=None):
        if n is None:
            n = self.batch_size
        metric = self.metrics[metric_name]
        metric['sum'] += value * n
        metric['count'] += n
        metric['avg'] = metric['sum'] / metric['count']

    def get_avg(self, metric_name):
        return self.metrics[metric_name]['avg']

    def get_metric_values(self):
        return [(metric, values['avg']) for metric, values in
                self.metrics.items()]

    def __str__(self):
        return ' | '.join(
            f'{metric_name} {metric["avg"]:.6f}' for metric_name, metric in
            self.metrics.items()
        )


def load_model(model_name, filename, cuda=True):
    model = make_model(model_name)
    model_state = torch.load(os.path.join(config.SAVE_MODEL_PATH, filename))
    model.load_state_dict(model_state)
    if cuda:
        model = model.cuda()
    return model


def save_model(model, filename):
    print(f'saving model {filename}')
    os.makedirs(config.SAVE_MODEL_PATH, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(config.SAVE_MODEL_PATH, filename))


def make_crayon_experiments(experiment_name, new=True):
    client = CrayonClient(hostname=config.CRAYON_HOSTNAME)
    train_experiment_name = f'{experiment_name}_train'
    valid_experiment_name = f'{experiment_name}_valid'
    if new:
        try:
            client.remove_experiment(train_experiment_name)
        except ValueError:
            pass
        try:
            client.remove_experiment(valid_experiment_name)
        except ValueError:
            pass
        train_experiment = client.create_experiment(train_experiment_name)
        train_experiment.scalar_steps['lr'] = 1
        valid_experiment = client.create_experiment(valid_experiment_name)
    else:
        train_experiment = client.open_experiment(train_experiment_name)
        valid_experiment = client.open_experiment(valid_experiment_name)
    return train_experiment, valid_experiment


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
