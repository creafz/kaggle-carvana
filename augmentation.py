from imgaug import augmenters as iaa

import config
import custom_transforms as t


augmentations = {
    'pad': (
        t.Pad.Both(size=(0, 0, 1, 1)),
        t.ExpandDims.Mask(axis=2),
    ),
    'crop': (
        t.RandomCrop.Both(original_size=(1918, 1280), crop_size=(1024, 1024)),
    ),
    'crop_fliplr_affine_color': (
        t.RandomCrop.Both(original_size=(1918, 1280), crop_size=(1024, 1024)),
        t.ExpandDims.Mask(axis=2),
        t.ImgAug.Both(iaa.Fliplr(0.5)),
        t.ImgAug.Both(
            iaa.Affine(
                scale={'x': (0.95, 1.05), 'y': (0.95, 1.05)},
                rotate=(-16, 16),
                shear=(-16, 16),
                order=[0, 1],
                mode='reflect'
            ),
            p=0.25,
        ),
        t.ImgAug.Image(
            iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace='RGB',
                                     to_colorspace='HSV'),
                iaa.WithChannels([0], iaa.Add((-30, 30))),
                iaa.ChangeColorspace(from_colorspace='HSV',
                                     to_colorspace='RGB')
            ]),
            p=0.25
        ),
        t.Clip.Image(),
    ),
    'crop_fliplr': (
        t.RandomCrop.Both(original_size=(1918, 1280), crop_size=(1024, 1024)),
        t.ExpandDims.Mask(axis=2),
        t.ImgAug.Both(iaa.Fliplr(0.5)),
    ),
    'pad_fliplr_affine_color': (
        t.Pad.Both(size=(0, 0, 1, 1)),
        t.ExpandDims.Mask(axis=2),
        t.ImgAug.Both(iaa.Fliplr(0.5)),
        t.ImgAug.Both(
            iaa.Affine(
                scale={'x': (0.95, 1.05), 'y': (0.95, 1.05)},
                rotate=(-16, 16),
                shear=(-16, 16),
                order=[0, 1],
                mode='reflect'
            ),
            p=0.25,
        ),
        t.ImgAug.Image(
            iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace='RGB',
                                     to_colorspace='HSV'),
                iaa.WithChannels([0], iaa.Add((-30, 30))),
                iaa.ChangeColorspace(from_colorspace='HSV',
                                     to_colorspace='RGB')
            ]),
            p=0.25
        ),
        t.Clip.Image(),
    ),
    'pad_fliplr': (
        t.Pad.Both(size=(0, 0, 1, 1)),
        t.ExpandDims.Mask(axis=2),
        t.ImgAug.Both(iaa.Fliplr(0.5)),
    ),
    'resize_512_fliplr_affine_color': (
        t.Resize.Both((512, 512)),
        t.ExpandDims.Mask(axis=2),
        t.ImgAug.Both(iaa.Fliplr(0.5)),
        t.ImgAug.Both(
            iaa.Affine(
                scale={'x': (0.95, 1.05), 'y': (0.95, 1.05)},
                rotate=(-16, 16),
                shear=(-16, 16),
                order=[0, 1],
                mode='reflect'
            ),
            p=0.25,
        ),
        t.ImgAug.Image(
            iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace='RGB',
                                     to_colorspace='HSV'),
                iaa.WithChannels([0], iaa.Add((-30, 30))),
                iaa.ChangeColorspace(from_colorspace='HSV',
                                     to_colorspace='RGB')
            ]),
            p=0.25
        ),
        t.Clip.Image(),
    ),
    'resize_512': (
        t.Resize.Both((512, 512)),
        t.ExpandDims.Mask(axis=2),
    ),

}


post_augmentation_transforms = (
    t.AsContiguousArray.Both(),
    t.ToTensor.Both(),
    t.Normalize.Image(
        mean=config.IMG_MEAN,
        std=config.IMG_STD,
    ),
)


def make_augmentation_transform(name):
    return t.Compose([
        *augmentations[name],
        *post_augmentation_transforms,
    ])
