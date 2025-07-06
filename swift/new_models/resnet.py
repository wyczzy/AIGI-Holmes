import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np

import sys
# sys.path.append("/path/to/your/project")
#
# from AIGC.networks.denoising_rgb import DenoiseNet
# from AIGC.models.network_dncnn import DnCNN
# from AIGC.models.modal_extract import ModalitiesExtractor
import torch
import torchvision.transforms as transforms
from collections import OrderedDict



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'mobilenet']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False, use_low_level="npr"):
        super(ResNet, self).__init__()

        self.unfoldSize = 2
        self.unfoldIndex = 0
        self.use_low_level = use_low_level
        assert self.unfoldSize > 1
        assert -1 < self.unfoldIndex and self.unfoldIndex < self.unfoldSize*self.unfoldSize
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 , layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.mlp = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.fc1 = nn.Linear(512, num_classes)
        # self.fc1 = nn.Linear(112, num_classes)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.denormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(self.mean, self.std)],
            std=[1 / s for s in self.std]

        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        if self.use_low_level != "npr":
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            self.model_weights_dncnn = '/path/to/model_zoo/dncnn_color_blind.pth'
            self.model_weights_noiseprint = '/path/to/model_zoo/dncnn_color_blind.pth'
            self.model_weights_lnp = '/path/to/weights/preprocessing/sidd_rgb.pth'
            self.MODALS = ['noiseprint', 'bayar', 'srm']

            self.model_lnp = DenoiseNet()
            load_checkpoint(self.model_lnp, self.model_weights_lnp)
            # print("===>Testing using weights: ", args.weights)
            self.model_lnp.cuda()
            self.model_lnp.eval()

            self.model_dncnn = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
            self.model_dncnn.load_state_dict(torch.load(self.model_weights_dncnn), strict=True)
            self.model_dncnn = self.model_dncnn.cuda()
            self.model_dncnn.eval()
            for k, v in self.model_dncnn.named_parameters():
                v.requires_grad = False

            self.modal_extractor = ModalitiesExtractor(self.MODALS, '/path/to/weights/noiseprint/np++.pth')
            if 'bayar' in self.MODALS:
                self.modal_extractor.load_state_dict(torch.load('/path/to/weights/modal_extractor/bayar_mhsa.pth'), strict=False)
                self.modal_extractor.bayar.eval()
                for param in self.modal_extractor.parameters():
                    param.requires_grad = False
            self.modal_extractor = self.modal_extractor.cuda()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest',
            recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)

    def forward(self, x):
        # n,c,w,h = x.shape
        # if -1*w%2 != 0: x = x[:,:,:w%2*-1,:      ]
        # if -1*h%2 != 0: x = x[:,:,:      ,:h%2*-1]
        # factor = 0.5
        # x_half = F.interpolate(x, scale_factor=factor, mode='nearest', recompute_scale_factor=True)
        # x_re   = F.interpolate(x_half, scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
        # NPR  = x - x_re
        if self.use_low_level == 'npr':
            n,c,w,h = x.shape
            if w%2 == 1 : x = x[:,:,:-1,:]
            if h%2 == 1 : x = x[:,:,:,:-1]
            NPR  = (x - self.interpolate(x, 0.5))*2/3
            x = NPR

        if self.use_low_level == 'dncnn':
            with torch.no_grad():
                dex = self.denormalize(x)

                rgb_restored = self.model_dncnn.noise(dex)
                rgb_restored = torch.clamp(rgb_restored, 0, 1)
                rgb_restored = (rgb_restored * 255).round()
                rgb_restored = rgb_restored.to(torch.uint8)
                img = rgb_restored*255
                img = img.to(torch.float32)/255
                x = self.normalize(img)

        if self.use_low_level == 'lnp':
            with torch.no_grad():
                dex = self.denormalize(x)

                rgb_restored = self.model_lnp(dex)
                rgb_restored = torch.clamp(rgb_restored, 0, 1)
                rgb_restored = (rgb_restored * 255).round()
                rgb_restored = rgb_restored.to(torch.uint8)
                img = rgb_restored * 255
                img = img.to(torch.float32) / 255
                x = self.normalize(img)

        if self.use_low_level == "noiseprint":
            with torch.no_grad():
                dex = self.denormalize(x)

                rgb_restored = self.modal_extractor(dex)
                rgb_restored = torch.clamp(rgb_restored[0], 0, 1)
                rgb_restored = (rgb_restored * 255).round()
                rgb_restored = rgb_restored.to(torch.uint8)
                img = rgb_restored * 255
                img = img.to(torch.float32) / 255
                x = self.normalize(img)

        if self.use_low_level == "bayar":
            with torch.no_grad():
                dex = self.denormalize(x)

                rgb_restored = self.modal_extractor(dex)
                rgb_restored = torch.clamp(rgb_restored[1], 0, 1)
                rgb_restored = (rgb_restored * 255).round()
                rgb_restored = rgb_restored.to(torch.uint8)
                img = rgb_restored * 255
                img = img.to(torch.float32) / 255
                x = self.normalize(img)

        if self.use_low_level == "srm":
            with torch.no_grad():
                dex = self.denormalize(x)

                rgb_restored = self.modal_extractor(dex)
                rgb_restored = torch.clamp(rgb_restored[2], 0, 1)
                rgb_restored = (rgb_restored * 255).round()
                rgb_restored = rgb_restored.to(torch.uint8)
                img = rgb_restored * 255
                img = img.to(torch.float32) / 255
                x = self.normalize(img)



            # n, c, w, h = x.shape
            # if w % 2 == 1: x = x[:, :, :-1, :]
            # if h % 2 == 1: x = x[:, :, :, :-1]
            # NPR = (x - self.interpolate(x, 0.5)) * 2 / 3
            # x = NPR

        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # denormalize = transforms.Normalize(
        #     mean=[-m / s for m, s in zip(mean, std)],
        #     std=[1 / s for s in std]
        #
        # )

        x = self.conv1(x)
        # x = self.conv1(NPR*2.0/3.0)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.mlp(x)
        # x = self.fc2(x)
        x = self.fc1(x)

        return x

    def forward_features(self, x):
        # n,c,w,h = x.shape
        # if -1*w%2 != 0: x = x[:,:,:w%2*-1,:      ]
        # if -1*h%2 != 0: x = x[:,:,:      ,:h%2*-1]
        # factor = 0.5
        # x_half = F.interpolate(x, scale_factor=factor, mode='nearest', recompute_scale_factor=True)
        # x_re   = F.interpolate(x_half, scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
        # NPR  = x - x_re
        if self.use_low_level == 'npr':
            n,c,w,h = x.shape
            if w%2 == 1 : x = x[:,:,:-1,:]
            if h%2 == 1 : x = x[:,:,:,:-1]
            NPR  = (x - self.interpolate(x, 0.5))*2/3
            x = NPR
        if self.use_low_level == 'dncnn':
            with torch.no_grad():
                dex = self.denormalize(x)
                rgb_restored = self.model_dncnn.noise(dex)
                rgb_restored = torch.clamp(rgb_restored, 0, 1)
                rgb_restored = (rgb_restored * 255).round()
                rgb_restored = rgb_restored.to(torch.uint8)
                img = rgb_restored*255
                img = img.to(torch.float32)/255
                x = self.normalize(img)
        if self.use_low_level == 'lnp':
            with torch.no_grad():
                dex = self.denormalize(x)
                rgb_restored = self.model_lnp(dex)
                rgb_restored = torch.clamp(rgb_restored, 0, 1)
                rgb_restored = (rgb_restored * 255).round()
                rgb_restored = rgb_restored.to(torch.uint8)
                img = rgb_restored * 255
                img = img.to(torch.float32) / 255
                x = self.normalize(img)
        if self.use_low_level == "noiseprint":
            with torch.no_grad():
                dex = self.denormalize(x)
                rgb_restored = self.modal_extractor(dex)
                rgb_restored = torch.clamp(rgb_restored[0], 0, 1)
                rgb_restored = (rgb_restored * 255).round()
                rgb_restored = rgb_restored.to(torch.uint8)
                img = rgb_restored * 255
                img = img.to(torch.float32) / 255
                x = self.normalize(img)
        if self.use_low_level == "bayar":
            with torch.no_grad():
                dex = self.denormalize(x)
                rgb_restored = self.modal_extractor(dex)
                rgb_restored = torch.clamp(rgb_restored[1], 0, 1)
                rgb_restored = (rgb_restored * 255).round()
                rgb_restored = rgb_restored.to(torch.uint8)
                img = rgb_restored * 255
                img = img.to(torch.float32) / 255
                x = self.normalize(img)
        if self.use_low_level == "srm":
            with torch.no_grad():
                dex = self.denormalize(x)
                rgb_restored = self.modal_extractor(dex)
                rgb_restored = torch.clamp(rgb_restored[2], 0, 1)
                rgb_restored = (rgb_restored * 255).round()
                rgb_restored = rgb_restored.to(torch.uint8)
                img = rgb_restored * 255
                img = img.to(torch.float32) / 255
                x = self.normalize(img)
            # n, c, w, h = x.shape
            # if w % 2 == 1: x = x[:, :, :-1, :]
            # if h % 2 == 1: x = x[:, :, :, :-1]
            # NPR = (x - self.interpolate(x, 0.5)) * 2 / 3
            # x = NPR
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # denormalize = transforms.Normalize(
        #     mean=[-m / s for m, s in zip(mean, std)],
        #     std=[1 / s for s in std]
        #
        # )
        x = self.conv1(x)
        # x = self.conv1(NPR*2.0/3.0)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        # x = self.fc2(x)
        # x = self.fc1(x)
        return x


class MobileNetV3_First6Layers(torch.nn.Module):
    def __init__(self, original_model, num_classes=1):
        super(MobileNetV3_First6Layers, self).__init__()
        self.features = torch.nn.Sequential(*list(original_model.features.children())[:12])
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(112, num_classes)

    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest',
            recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)

    def forward(self, x):
        n, c, w, h = x.shape
        if w % 2 == 1: x = x[:, :, :-1, :]
        if h % 2 == 1: x = x[:, :, :, :-1]
        NPR = (x - self.interpolate(x, 0.5)) * 2 / 3
        x = NPR
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def mobilenet(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    import torch
    import torchvision.models as models

    # 调用 MobileNetV3 Large 版本
    model = models.mobilenet_v3_large(pretrained=pretrained)
    model_first6 = MobileNetV3_First6Layers(model, **kwargs)

    # model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model_first6
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
