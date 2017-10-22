import os

import cv2
import numpy as np
from skimage.io import imread
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from torch.utils.data import Dataset

import config


class CarvanaTrainDataset(Dataset):
    def __init__(self, mode, folds, fold_num, transform=None):
        assert mode in {'train', 'valid'}
        self.mode = mode
        self.transform = transform
        files = list(sorted(os.listdir(config.TRAIN_IMAGES_HQ_PATH)))
        groups = [name.split('_')[0] for name in files]
        files, groups = shuffle(files, groups, random_state=config.SEED)
        files, groups = np.array(files), np.array(groups)
        group_kfold = GroupKFold(n_splits=folds)
        train_index, valid_index = \
            list(group_kfold.split(files, groups=groups))[fold_num]
        assert len(set(groups[train_index]) & set(groups[valid_index])) == 0
        self.dataset_index = train_index if self.mode == 'train' \
            else valid_index
        self.files = files

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index):
        i = self.dataset_index[index]
        filename = self.files[i]
        image = cv2.imread(os.path.join(config.TRAIN_IMAGES_HQ_PATH, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = imread(
            os.path.join(
                config.TRAIN_MASKS_PATH,
                filename.replace('.jpg', '_mask.gif')
            )
        )

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


class CarvanaTestDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        files = [
            os.path.join(config.TEST_IMAGES_PATH, filename)
            for filename in sorted(os.listdir(config.TEST_IMAGES_PATH))
        ]
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filepath = self.files[i]
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, filepath
