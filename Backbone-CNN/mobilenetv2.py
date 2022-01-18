"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import re
import torch
import torch.nn as nn
import math

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, (int(v + divisor / 2) // divisor) * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv3x3_block(in_c, out_c, stride):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True)
    )

def conv1x1_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True)
    )

class inverted_residual(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super(inverted_residual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(in_c * expand_ratio)
        self.idetity = (stride == 1) and (in_c == out_c)

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pointwise linear
                nn.Conv2d(hidden_dim, out_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.conv = nn.Sequential(
                # pointwise
                nn.Conv2d(in_c, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pointwise linear
                nn.Conv2d(hidden_dim, out_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_c)
            )
    
    def forward(self, x):
        if self.idetity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobilenetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1):
        super(MobilenetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        in_channel = _make_divisible(32*width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv3x3_block(3, in_channel, stride=2)]
        # building inverted residual blocks
        block = inverted_residual
        for t, c, n, s in self.cfgs:
            out_channel = _make_divisible(c*width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(in_channel, out_channel, s if i==0 else 1, t))
                in_channel = out_channel
        
        self.features = nn.Sequential(*layers)
        # building last several layers
        out_channel = _make_divisible(1280*width_mult, 4 if width_mult==0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv1x1_block(in_channel, out_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(out_channel, num_classes)

        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(num_classes=10, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobilenetV2(num_classes=10, **kwargs)
    
# def test():
#     net = mobilenetv2()
#     print(net)
#     data = torch.rand(4, 3, 224, 224)
#     output = net(data)
#     print(output.size())

# test()