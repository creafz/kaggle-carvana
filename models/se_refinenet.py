import torch.nn as nn
import torchvision.models as models

from models.layers import SqueezeExcitation


'''
RefineNet: Multi-Path Refinement Networks for High-Resolution
Semantic Segmentation
https://arxiv.org/abs/1611.06612

Based on this implementation of the paper:
https://github.com/thomasjpfan/pytorch_refinenet
'''


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.bn1 = nn.BatchNorm2d(features)
        self.bn2 = nn.BatchNorm2d(features)

        self.sqex = SqueezeExcitation(features)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.sqex(out)
        out = out + x
        return out


class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()

        _, max_size = max(shapes, key=lambda x: x[1])

        for i, shape in enumerate(shapes):
            feat, size = shape
            if max_size % size != 0:
                raise ValueError(f'max_size not divisble by shape {i}')

            scale_factor = max_size // size
            if scale_factor != 1:
                self.add_module(f'resolve{i}', nn.Sequential(
                    nn.Conv2d(feat, out_feats, kernel_size=3,
                              stride=1, padding=1, bias=False),
                    nn.Upsample(scale_factor=scale_factor, mode='bilinear')
                ))
            else:
                self.add_module(
                    f'resolve{i}',
                    nn.Conv2d(feat, out_feats, kernel_size=3,
                              stride=1, padding=1, bias=False)
                )

    def forward(self, *xs):

        output = self.resolve0(xs[0])

        for i, x in enumerate(xs[1:], 1):
            output = output + self.__getattr__(f'resolve{i}')(x)

        return output


class ChainedResidualPool(nn.Module):
    def __init__(self, feats):
        super().__init__()
        self.bn = nn.BatchNorm2d(feats)

        self.relu = nn.ReLU(inplace=False)
        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1,
                      bias=False)
        )
        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1,
                      bias=False)
        )
        self.block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1,
                      bias=False)
        )
        self.sqex = SqueezeExcitation(feats)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)

        out = self.block1(x)
        x = x + out
        out = self.block2(out)
        x = x + out
        out = self.block3(out)

        out = self.sqex(out)
        x = x + out

        return x


class RefineNetBlock(nn.Module):
    def __init__(self, features, *shapes):
        super().__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module(f'rcu{i}', nn.Sequential(
                ResidualConvUnit(feats),
                ResidualConvUnit(feats)
            ))

        if len(shapes) != 1:
            self.mrf = MultiResolutionFusion(features, *shapes)
        else:
            self.mrf = None

        self.crp = ChainedResidualPool(features)
        self.output_conv = ResidualConvUnit(features)

    def forward(self, *xs):
        for i, x in enumerate(xs):
            x = self.__getattr__(f'rcu{i}')(x)

        if self.mrf is not None:
            out = self.mrf(*xs)
        else:
            out = xs[0]

        out = self.crp(out)
        return self.output_conv(out)


class SERefineNet(nn.Module):
    def __init__(self, input_shape,
                 num_classes=1,
                 features=48,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True,
                 weight_init=None,
                 ):
        super().__init__()

        input_channel, input_size = input_shape

        if input_size % 32 != 0:
            raise ValueError(f'{input_shape} not divisble by 32')

        resnet = resnet_factory(pretrained=pretrained)

        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze_resnet:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            2048, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refinenet4 = RefineNetBlock(
            2 * features, (2 * features, input_size // 32))
        self.refinenet3 = RefineNetBlock(
            features, (2 * features, input_size // 32),
            (features, input_size // 16))
        self.refinenet2 = RefineNetBlock(
            features, (features, input_size // 16),
            (features, input_size // 8))
        self.refinenet1 = RefineNetBlock(
            features, (features, input_size // 8), (features, input_size // 4))

        self.output_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualConvUnit(features),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualConvUnit(features),
            nn.Conv2d(features, num_classes, kernel_size=1, stride=1,
                      padding=0, bias=True)
        )
        if weight_init is not None:
            for i in range(1, 5):
                weight_init(self.__getattr__(f'layer{i}_rn'))
                weight_init(self.__getattr__(f'refinenet{i}'))
            weight_init(self.output_conv)

    def forward(self, x):

        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out = self.output_conv(path_1)

        return out
