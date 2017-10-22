import itertools

import cv2
from imgaug import augmenters as iaa
import numpy as np
import torch
import torchvision


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *a):
        for t in self.transforms:
            a = t(*a)
        return a


class TestCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, a):
        for t in self.transforms:
            a = t(a)
        return a


class CustomTransform:
    @classmethod
    def Both(cls, *a, **kw):
        return cls(*a, **kw)._set_mode('both')

    @classmethod
    def Image(cls, *a, **kw):
        return cls(*a, **kw)._set_mode('image')

    @classmethod
    def AllImages(cls, *a, **kw):
        return cls(*a, **kw)._set_mode('all_images')

    @classmethod
    def Mask(cls, *a, **kw):
        return cls(*a, **kw)._set_mode('mask')

    def _set_mode(self, mode):
        assert mode in {'image', 'mask', 'both', 'all_images'}
        self.mode = mode
        return self

    def _pre_call_hook(self):
        pass

    def __call__(self, *a, **kw):
        self._pre_call_hook()
        if self.mode == 'all_images':
            return self.batch_input_transform(*a, **kw)
        else:
            return self.single_input_transform(*a, **kw)

    def batch_input_transform(self, images):
        return [self.transform_batch_image(image, index) for index, image in
                enumerate(images)]

    def single_input_transform(self, image, mask=None):
        if self.mode in {'both', 'image'}:
            image = self.transform_image(image)

        if self.mode in {'both', 'mask'}:
            mask = self.transform_mask(mask)

        return image if mask is None else (image, mask)

    def transform(self, image):
        raise NotImplementedError()

    def transform_batch_image(self, image, index):
        return self.transform_image(image)

    def transform_image(self, image):
        return self.transform(image)

    def transform_mask(self, mask):
        return self.transform(mask)


class ToTensor(CustomTransform):
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def transform(self, image):
        return self.to_tensor(image).type(torch.FloatTensor)


class Normalize(CustomTransform):
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean, std)

    def transform(self, image):
        return self.normalize(image)


class Pad(CustomTransform):
    def __init__(self, size=(0, 0, 0, 0), type=cv2.BORDER_REFLECT):
        self.size = size
        self.type = type

    def transform(self, image):
        return cv2.copyMakeBorder(image, *self.size, self.type)


class Resize(CustomTransform):
    def __init__(self, output_size):
        self.output_size = output_size

    def transform(self, image):
        return cv2.resize(image, self.output_size)


class Lambda(CustomTransform):
    def __init__(self, func):
        self.func = func

    def transform(self, image):
        return self.func(image)


class ExpandDims(CustomTransform):
    def __init__(self, axis):
        self.axis = axis

    def transform(self, image):
        return np.expand_dims(image, self.axis)


class ImgAug(CustomTransform):
    def __init__(self, augmenters, p=None):
        if not isinstance(augmenters, list):
            augmenters = [augmenters]
        seq = iaa.Sequential(augmenters)
        if p is not None:
            seq = iaa.Sometimes(p, seq)
        self.seq = seq
        self.seq_det = None

    def _pre_call_hook(self):
        self.seq_det = self.seq.to_deterministic()

    def transform(self, image):
        return self.seq_det.augment_image(image)


class FlipRotate(CustomTransform):
    def __init__(self):
        flip_options = (False, True)
        rotate_options = range(4)
        self.options = list(itertools.product(flip_options, rotate_options))

    def _pre_call_hook(self):
        option_index = np.random.randint(0, len(self.options))
        self.option = self.options[option_index]

    def transform(self, image):
        flip, rotate_k = self.option
        if flip:
            image = np.fliplr(image)
        image = np.rot90(image, rotate_k)
        return image


class RandomCrop(CustomTransform):
    def __init__(self, original_size, crop_size):
        self.original_size = original_size
        self.crop_size = crop_size

    def _pre_call_hook(self):
        original_width, original_height = self.original_size
        width, height = self.crop_size
        x = np.random.randint(0, original_width - width - 1)
        y = np.random.randint(0, original_height - height - 1)
        self.crop_coords = np.s_[y: y + height, x: x + width]

    def transform(self, image):
        return image[self.crop_coords]


class Clip(CustomTransform):
    def __init__(self, min_value=0, max_value=255):
        self.min_value = min_value
        self.max_value = max_value

    def transform(self, image):
        return np.clip(image, self.min_value, self.max_value)


class AsContiguousArray(CustomTransform):
    def transform(self, image):
        return np.ascontiguousarray(image)
