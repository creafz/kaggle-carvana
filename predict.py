import os

import click
from click import option as opt
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import config
import custom_transforms as t
import dataset
from utils import load_model


def save_predictions_to_png(model, fold_predictions_dir, flip_lr=False):
    transform = t.TestCompose([
        t.Pad.Image(size=(0, 0, 1, 1)),
        t.Lambda.Image(lambda img: np.fliplr(img) if flip_lr else img),
        t.AsContiguousArray.Image(),
        t.ToTensor.Image(),
        t.Normalize.Image(config.IMG_MEAN, config.IMG_STD),
    ])
    test_dataset = dataset.CarvanaTestDataset(transform=transform)
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    for image_index, (image, filenames) in tqdm(enumerate(test_data_loader),
                                                total=len(test_data_loader)):
        filename = os.path.splitext(os.path.basename(filenames[0]))[0]
        image = Variable(image, volatile=True).cuda(async=True)
        y_pred = torch.sigmoid(model(image)).data.cpu().numpy()
        y_pred = y_pred.squeeze((0, 1))
        predictions = np.uint8(y_pred * 255)
        predictions = predictions[:, 1:-1]

        if flip_lr:
            filename += '_flip_lr'

        cv2.imwrite(
            os.path.join(fold_predictions_dir, f'{filename}.png'),
            predictions)


@click.command()
@opt('--dirname', type=str, required=True)
@opt('--model-name', type=str, required=True)
@opt('--experiment-name', type=str, required=True)
@opt('--folds', default=6)
@opt('--fold-num', default=0)
def main(dirname, model_name, experiment_name, folds, fold_num):
    os.makedirs(config.PREDICTIONS_PATH, exist_ok=True)
    predictions_dir = os.path.join(config.PREDICTIONS_PATH, dirname)
    os.makedirs(predictions_dir, exist_ok=True)
    fold_predictions_dir = os.path.join(predictions_dir, f'fold_{fold_num}_{folds}')
    os.makedirs(fold_predictions_dir, exist_ok=True)
    print(fold_predictions_dir)
    weights_file = f'{experiment_name}_{fold_num}_{folds}_best.pth'
    model = load_model(model_name, weights_file).eval()

    for flip_lr in (False, True):
        save_predictions_to_png(model, fold_predictions_dir, flip_lr)


if __name__ == '__main__':
    main()
