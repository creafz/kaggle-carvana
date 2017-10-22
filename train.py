import os

import click
from click import option as opt
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from augmentation import make_augmentation_transform, augmentations
import config
import dataset
from loss import dice_coef, losses
from models.make_model import make_model
from optimizer import get_optimizer
from utils import save_model, load_model, \
    MetricMonitor, make_crayon_experiments, set_seed


def forward_pass(images, masks, model, loss_fn, epoch, stream, monitor,
                 mode='train'):
    volatile = mode != 'train'
    images = Variable(images, volatile=volatile).cuda(async=True)
    masks = Variable(masks, volatile=volatile).cuda(async=True)
    outputs = model(images)
    loss = loss_fn(outputs, masks)
    dice_thresholded = dice_coef(F.sigmoid(outputs), masks, threshold=0.5)
    dice = dice_coef(F.sigmoid(outputs), masks)
    monitor.update('loss', loss.data[0])
    monitor.update('dice_thresholded', dice_thresholded.data[0])
    monitor.update('dice', dice.data[0])
    stream.set_description(
        f'epoch: {epoch} | '
        f'{mode}: {monitor}'
    )
    return loss, outputs


def train(train_data_loader, model, optimizer, iter_size, loss_fn, epoch,
          experiment):
    model.train()

    train_monitor = MetricMonitor(batch_size=train_data_loader.batch_size)
    stream = tqdm(train_data_loader)

    for i, (images, masks) in enumerate(stream, start=1):
        loss, _ = forward_pass(images, masks, model, loss_fn, epoch, stream,
                               train_monitor,
                               mode='train')
        loss.backward()

        if i % iter_size == 0 or i == len(train_data_loader):
            optimizer.step()
            optimizer.zero_grad()

    experiment.add_scalar_value('optimizer/lr',
                                optimizer.param_groups[0]['lr'], step=epoch)
    for metric, value in train_monitor.get_metric_values():
        experiment.add_scalar_value(f'metric/{metric}', value, step=epoch)


def validate(valid_data_loader, model, loss_fn, epoch, experiment,
             valid_predictions_dir, save_validation_image_probability):
    model.eval()

    valid_monitor = MetricMonitor(batch_size=valid_data_loader.batch_size)
    stream = tqdm(valid_data_loader)

    valid_predictions_dir_for_epoch = os.path.join(valid_predictions_dir,
                                                   str(epoch))
    os.makedirs(valid_predictions_dir_for_epoch, exist_ok=True)

    i = 0
    for (images, masks) in stream:
        _, outputs = forward_pass(images, masks, model, loss_fn, epoch, stream,
                                  valid_monitor, mode='valid')

        for image, mask in zip(images, outputs):
            if np.random.random() > save_validation_image_probability:
                continue
            y_pred = torch.sigmoid(mask).data.cpu().numpy()
            y_pred = y_pred.squeeze(0)
            image = image.numpy().transpose(1, 2, 0)
            for channel, (mean, std) in enumerate(
                    zip(config.IMG_MEAN, config.IMG_STD)):
                image[:, :, channel] = image[:, :, channel] * std + mean

            image = (np.uint8(image * 255))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(valid_predictions_dir_for_epoch,
                                     f'{i}_image.png'), image)
            cv2.imwrite(os.path.join(valid_predictions_dir_for_epoch,
                                     f'{i}_mask.png'),
                        (np.uint8(y_pred * 255)))
            i += 1

    for metric, value in valid_monitor.get_metric_values():
        experiment.add_scalar_value(f'metric/{metric}', value, step=epoch)

    return valid_monitor


def train_and_validate(
        train_data_loader,
        valid_data_loader,
        model,
        optimizer,
        iter_size,
        scheduler,
        loss_fn,
        epochs,
        experiment_name,
        save_best_model,
        experiments,
        valid_predictions_dir,
        save_validation_image_probability,
        start_epoch,
):
    train_experiment, valid_experiment = experiments
    best_val_metric = float('+inf')

    model.train()
    print('starting train...')
    for epoch in range(start_epoch, epochs + 1):
        train(train_data_loader, model, optimizer, iter_size, loss_fn, epoch,
              train_experiment)
        val_monitor = validate(valid_data_loader, model, loss_fn, epoch,
                               valid_experiment, valid_predictions_dir,
                               save_validation_image_probability)
        val_metric = val_monitor.get_avg('loss')
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            if save_best_model:
                print('saving best model...')
                save_model(model, f'{experiment_name}_best.pth')

        scheduler.step(val_metric)

    return model


@click.command()
@opt('--batch-size', default=1)
@opt('--optimizer-name', type=click.Choice(['adam', 'sgd']), default='adam')
@opt('--lr', default=1e-4)
@opt('--epochs', default=100)
@opt('--iter-size', default=32)
@opt('--save-best-model', default=True)
@opt('--model-name', type=str, required=True)
@opt('--experiment-name', type=str, required=True)
@opt('--load-best-model', is_flag=True)
@opt('--folds', default=6)
@opt('--fold-num', default=0)
@opt('--start-epoch', default=1)
@opt('--save-validation-image-probability', default=0.025)
@opt('--create-new-experiment', is_flag=True)
@opt('--augmentation-train', type=click.Choice(augmentations.keys()),
     default='crop_fliplr_affine_color')
@opt('--augmentation-valid', type=click.Choice(augmentations.keys()),
     default='crop_fliplr')
@opt('--loss', type=click.Choice(losses.keys()), default='bce_dice')
def main(
        batch_size,
        optimizer_name,
        lr,
        epochs,
        iter_size,
        save_best_model,
        model_name,
        experiment_name,
        load_best_model,
        folds,
        fold_num,
        start_epoch,
        save_validation_image_probability,
        create_new_experiment,
        augmentation_train,
        augmentation_valid,
        loss,
):
    set_seed(config.SEED)
    transform_train = make_augmentation_transform(augmentation_train)
    transform_valid = make_augmentation_transform(augmentation_valid)
    full_experiment_name = f'{experiment_name}_{fold_num}_{folds}'
    print(full_experiment_name)

    if load_best_model:
        model = load_model(model_name, f'{full_experiment_name}_best.pth')
    else:
        model = make_model(model_name)

    new_experiment = not load_best_model or create_new_experiment
    experiments = make_crayon_experiments(full_experiment_name,
                                          new=new_experiment)
    valid_predictions_dir = os.path.join(config.VALID_PREDICTIONS_PATH,
                                         full_experiment_name)
    os.makedirs(valid_predictions_dir, exist_ok=True)

    loss_fn = losses[loss]()
    optimizer = get_optimizer(optimizer_name, lr,
                              (p for p in model.parameters() if
                               p.requires_grad))
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=3,
        verbose=True,
        threshold=1e-5,
        min_lr=0,
        mode='min',
    )

    dataset_params = {
        'folds': folds,
        'fold_num': fold_num,

    }
    train_dataset = dataset.CarvanaTrainDataset(
        **dataset_params,
        mode='train',
        transform=transform_train,
    )

    valid_dataset = dataset.CarvanaTrainDataset(
        **dataset_params,
        mode='valid',
        transform=transform_valid,
    )

    data_loader_args = {
        'pin_memory': True,
        'num_workers': config.NUM_WORKERS,
    }

    train_data_loader = DataLoader(
        **data_loader_args,
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_data_loader = DataLoader(
        **data_loader_args,
        dataset=valid_dataset,
        batch_size=batch_size,
    )

    train_and_validate(
        train_data_loader,
        valid_data_loader,
        model,
        optimizer,
        iter_size,
        scheduler,
        loss_fn,
        epochs,
        full_experiment_name,
        save_best_model,
        experiments,
        valid_predictions_dir,
        save_validation_image_probability,
        start_epoch,
    )


if __name__ == '__main__':
    main()
