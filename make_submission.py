from io import StringIO
import os
import zipfile

import click
from click import option as opt
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

import config


# RLE Encoding functions from https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


@click.command()
@opt('--dirname', type=str, required=True)
@opt('--threshold', default=127)
@opt('--submission-filename', type=str, required=True)
def main(dirname, threshold, submission_filename):
    predictions_parent_dir = os.path.join(config.PREDICTIONS_PATH, dirname)

    submission = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    submission.set_index('img', inplace=True)

    fold_dirs = os.listdir(predictions_parent_dir)
    filenames = os.listdir(os.path.join(predictions_parent_dir, fold_dirs[0]))
    unique_filenames = list(
        sorted({filename.replace('_flip_lr', '') for filename in filenames}))

    for filename in tqdm(unique_filenames):
        predictions = np.zeros((len(fold_dirs) * 2, 1280, 1918))
        for i, fold_dir in enumerate(fold_dirs):
            img_file = os.path.join(predictions_parent_dir, fold_dir, filename)
            flip_lr_img_file = os.path.join(predictions_parent_dir, fold_dir,
                                            filename.replace('.png',
                                                             '_flip_lr.png'))
            predictions[i * 2] = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            predictions[i * 2 + 1] = np.fliplr(
                cv2.imread(flip_lr_img_file, cv2.IMREAD_UNCHANGED))
        predictions = np.uint8(np.mean(predictions, axis=0))
        predictions[predictions < threshold] = 0
        predictions[predictions >= threshold] = 255
        rle_mask = rle_encode(predictions)
        img_name = filename.replace('.png', '.jpg')
        submission.loc[img_name]['rle_mask'] = rle_to_string(rle_mask)

    buffer = StringIO()
    submission.reset_index(inplace=True)
    submission.to_csv(buffer, index=False)
    buffer.seek(0)
    os.makedirs(config.SUBMISSIONS_PATH, exist_ok=True)
    submission_path = os.path.join(config.SUBMISSIONS_PATH,
                                   submission_filename + '.zip')
    with zipfile.ZipFile(submission_path, mode='w',
                         compression=zipfile.ZIP_DEFLATED) as f:
        f.writestr(submission_filename, buffer.read())


if __name__ == '__main__':
    main()
