from torch.optim import SGD, Adam


def get_optimizer(optimizer_name, lr, model_params):
    print(f'Using {optimizer_name} optimizer')
    if optimizer_name == 'sgd':
        return SGD(model_params, lr=lr, weight_decay=1e-6, momentum=0.9, nesterov=True)
    else:
        return Adam(model_params, lr=lr)
